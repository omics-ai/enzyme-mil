"""Model definitions for the three released enzyme-MIL checkpoints.

Each enzyme is a *bag* and its DCTdomain domains are *instances*. A gated-attention pooler turns
the bag into one fixed-width representation, and a classifier head emits independent logits over a
fixed vocabulary of 5,242 EC numbers.

Two architectures live here because the three checkpoints come from two training generations:

``AttentionMIL``
    ``milbc`` and ``milbc_plus``. Reads the local domain tokens. ``milbc_plus`` concatenates the
    whole-sequence ESM2 embedding onto the pooled bag representation before the classifier.

``JointMILCL``
    ``milcl_plus``. Reads the same tokens plus the whole-protein DCT vector as a seventeenth,
    fuses ESM2 into every token before pooling, and carries a 256-dimensional contrastive
    projection alongside the classifier. Max-separation inference compares that projection against
    the split100 EC centroids in ``checkpoints/milcl_plus/centroids.npy``.

The geometry of each checkpoint is fixed; ``load_state_dict`` runs strict, so a mismatch raises
rather than loading silently wrong weights.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

NUM_CLASSES = 5242
DOMAIN_DIM = 480
ESM_DIM = 1280

# Geometry of the MILCL backbone, taken from the MILBC+ checkpoint it was initialised from.
MILCL_CONFIG = {
    "instance_hidden_dim": 512,
    "attention_hidden_dim": 256,
    "esm_hidden_dim": 512,
    "classifier_hidden_dim": 512,
    "projection_hidden_dim": 512,
    "projection_dim": 256,
    "dropout": 0.2,
    "projection_dropout": 0.1,
}


class AttentionMIL(nn.Module):
    """Gated-attention MIL over 16 local domain tokens.

    ``global_input_dim == 0`` is the MILBC form. A positive value is the MILBC+ form: the
    whole-sequence ESM2 embedding is concatenated onto the pooled bag representation, so the
    domain branch and the sequence branch stay independent until the classifier.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        instance_input_dim: int = DOMAIN_DIM,
        instance_embedding_dim: int = 480,
        attention_hidden_dim: int = 320,
        global_input_dim: int = 0,
        classifier_hidden_dim: int = 320,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.global_input_dim = global_input_dim
        self.instance_encoder = nn.Sequential(
            nn.Linear(instance_input_dim, instance_embedding_dim),
            nn.LayerNorm(instance_embedding_dim),
            nn.ReLU(),
        )
        self.attention_V = nn.Sequential(
            nn.Linear(instance_embedding_dim, attention_hidden_dim),
            nn.LayerNorm(attention_hidden_dim),
            nn.Tanh(),
        )
        self.attention_gate = nn.Sequential(
            nn.Linear(instance_embedding_dim, attention_hidden_dim),
            nn.LayerNorm(attention_hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(attention_hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(instance_embedding_dim + global_input_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(
        self,
        instances_in_bag: Tensor,
        mask: Tensor | None = None,
        global_feature: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        h_k = self.instance_encoder(instances_in_bag)
        u_k = self.attention_V(h_k) * self.attention_gate(h_k)
        e_k = self.attention_w(u_k).squeeze(-1)
        if mask is not None:
            e_k = e_k.masked_fill(mask == 0, -1e9)
        alpha_k = torch.softmax(e_k, dim=1)
        bag_representation = torch.sum(alpha_k.unsqueeze(-1) * h_k, dim=1)
        if global_feature is not None:
            bag_representation = torch.cat((bag_representation, global_feature), dim=1)
        return self.classifier(bag_representation), bag_representation


class GatedAttentionPooler(nn.Module):
    """Gated attention over the 17 domain tokens, concatenated with their mean and max."""

    def __init__(self, input_dim: int, hidden_dim: int, attention_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dim * 3
        self.input_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attention_v = nn.Linear(hidden_dim, attention_dim)
        self.attention_u = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)

    def forward(self, tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_dim:
            raise ValueError(f"tokens must have shape (batch, instances, {self.input_dim})")
        if mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("domain mask must be boolean with shape (batch, instances)")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("each bag must contain at least one valid domain")
        encoded = self.encoder(self.input_norm(tokens))
        gated = torch.tanh(self.attention_v(encoded)) * torch.sigmoid(self.attention_u(encoded))
        scores = self.attention_w(gated).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1) * mask.to(scores.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True)
        pooled = torch.sum(encoded * attention.unsqueeze(-1), dim=1)
        mask_values = mask.unsqueeze(-1)
        mean = (encoded * mask_values).sum(dim=1) / mask_values.sum(dim=1)
        maximum = encoded.masked_fill(~mask_values, torch.finfo(encoded.dtype).min).max(dim=1).values
        return torch.cat((pooled, mean, maximum), dim=-1), attention


class MILBCPlusBackbone(nn.Module):
    """The MILCL backbone: ESM2 is encoded once and fused into every domain token before pooling."""

    def __init__(self) -> None:
        super().__init__()
        config = MILCL_CONFIG
        self.esm_encoder = nn.Sequential(
            nn.LayerNorm(ESM_DIM),
            nn.Linear(ESM_DIM, config["esm_hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
        )
        self.pooler = GatedAttentionPooler(
            DOMAIN_DIM + config["esm_hidden_dim"],
            config["instance_hidden_dim"],
            config["attention_hidden_dim"],
            config["dropout"],
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.pooler.output_dim),
            nn.Linear(self.pooler.output_dim, config["classifier_hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["classifier_hidden_dim"], NUM_CLASSES),
        )

    def encode(self, domains: Tensor, domain_mask: Tensor, esm2: Tensor) -> tuple[Tensor, Tensor]:
        if esm2.ndim != 2 or esm2.shape[0] != domains.shape[0] or esm2.shape[1] != ESM_DIM:
            raise ValueError(f"esm2 must have shape (batch, {ESM_DIM})")
        esm_representation = self.esm_encoder(esm2)
        repeated = esm_representation.unsqueeze(1).expand(-1, domains.shape[1], -1)
        representation, attention = self.pooler(torch.cat((domains, repeated), dim=-1), domain_mask)
        return representation, attention

    def classify(self, representation: Tensor) -> Tensor:
        return self.classifier(representation)


class JointProjection(nn.Sequential):
    """Contrastive head. Its output is what max-separation compares against the EC centroids."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )


class JointMILCL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = MILBCPlusBackbone()
        self.projection = JointProjection(
            self.backbone.pooler.output_dim,
            MILCL_CONFIG["projection_hidden_dim"],
            MILCL_CONFIG["projection_dim"],
            MILCL_CONFIG["projection_dropout"],
        )

    def forward(self, domains: Tensor, domain_mask: Tensor, esm2: Tensor) -> Tensor:
        """The 256-dimensional contrastive embedding of every enzyme in the batch."""
        representation, _ = self.backbone.encode(domains, domain_mask, esm2)
        return self.projection(representation)


def load_milbc(checkpoint: str, with_esm: bool, device: str = "cpu") -> AttentionMIL:
    """Rebuild ``milbc`` (``with_esm=False``) or ``milbc_plus`` (``with_esm=True``)."""
    model = AttentionMIL(global_input_dim=ESM_DIM if with_esm else 0)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval().to(device)


def load_milcl(checkpoint: str, centroids: str, device: str = "cpu") -> tuple[JointMILCL, np.ndarray]:
    """Rebuild ``milcl_plus`` and read the EC centroids max-separation scores against."""
    model = JointMILCL()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state"])
    centers = np.load(centroids, allow_pickle=False)
    if centers.shape != (NUM_CLASSES, MILCL_CONFIG["projection_dim"]):
        raise ValueError(f"centroids must be ({NUM_CLASSES}, {MILCL_CONFIG['projection_dim']}), found {centers.shape}")
    return model.eval().to(device), centers
