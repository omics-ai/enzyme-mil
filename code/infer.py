#!/usr/bin/env python
"""Call EC numbers for a feature archive built by ``process.py``.

    python infer.py --npzdata ../example/example.npz \
                    --model milbc_plus --checkpoint ../checkpoints/milbc_plus/model_v1.pth

Writes ``../outputs/example_milbc_plus.csv`` -- the archive's stem and the model name -- unless
``--output`` says otherwise. Predictions come from a fixed vocabulary of 5,242 EC numbers learned
from split100, so an enzyme whose true EC is outside that vocabulary can never be called correctly.

Three models:
``milbc``       domain embeddings only, sigmoid >= 0.5
``milbc_plus``  domain embeddings + whole-sequence ESM2, sigmoid >= 0.5
``milcl_plus``  end-to-end contrastive model, CLEAN maximum separation
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from models import NUM_CLASSES, load_milbc, load_milcl

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "milbc": {"family": "milbc", "esm": False, "rule": "sigmoid"},
    "milbc_plus": {"family": "milbc", "esm": True, "rule": "sigmoid"},
    "milcl_plus": {"family": "milcl", "esm": True, "rule": "maxsep"},
}


def load_features(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """``(entries, domains, masks, esm2)`` from a ``process.py`` archive."""
    with np.load(path, allow_pickle=True) as archive:
        if "dataset_samples" not in archive.files:
            raise SystemExit(f"{path} must contain a 'dataset_samples' array; rebuild it with process.py")
        samples = archive["dataset_samples"]
    if samples.ndim != 1 or len(samples) == 0:
        raise SystemExit(f"{path}: dataset_samples must be a non-empty vector")

    first = samples[0]
    token_count, width = np.asarray(first["dct_feature"]).shape
    local_dim = np.asarray(first["dct_global"]).size
    esm_dim = np.asarray(first["esm2_feature"]).size
    if width not in {local_dim, 2 * local_dim}:
        raise SystemExit(f"{path}: DCT token width {width} must be {local_dim} or {2 * local_dim}")

    domains = np.zeros((len(samples), token_count + 1, local_dim), dtype=np.float32)
    masks = np.zeros((len(samples), token_count + 1), dtype=bool)
    esm2 = np.zeros((len(samples), esm_dim), dtype=np.float32)
    entries: list[str] = []
    seen: set[str] = set()
    for row, sample in enumerate(samples):
        entry = str(sample["entry"])
        if entry in seen:
            raise SystemExit(f"{path}: duplicate entry {entry}")
        seen.add(entry)
        feature = np.asarray(sample["dct_feature"], dtype=np.float32)
        whole = np.asarray(sample["dct_global"], dtype=np.float32).reshape(-1)
        mask = np.asarray(sample["dct_mask"], dtype=bool)
        embedding = np.asarray(sample["esm2_feature"], dtype=np.float32).reshape(-1)
        if feature.shape != (token_count, width) or mask.shape != (token_count,):
            raise SystemExit(f"{path}: inconsistent DCT shape for {entry}")
        if whole.shape != (local_dim,) or embedding.shape != (esm_dim,):
            raise SystemExit(f"{path}: inconsistent global or ESM shape for {entry}")
        if not mask.any():
            raise SystemExit(f"{path}: {entry} has no valid domain")
        if not (np.isfinite(feature).all() and np.isfinite(whole).all() and np.isfinite(embedding).all()):
            raise SystemExit(f"{path}: non-finite feature for {entry}")
        domains[row, :token_count] = feature[:, :local_dim]
        domains[row, -1] = whole
        masks[row, :token_count] = mask
        masks[row, -1] = True
        esm2[row] = embedding
        entries.append(entry)
    return entries, domains, masks, esm2


@torch.inference_mode()
def score_sigmoid(model, domains, masks, esm2, use_esm: bool, device: str, batch_size: int) -> np.ndarray:
    """Classifier logits, in archive order. ``sigmoid >= 0.5`` is the same test as ``logit >= 0``."""
    logits = np.zeros((len(domains), NUM_CLASSES), dtype=np.float32)
    for start in range(0, len(domains), batch_size):
        stop = min(start + batch_size, len(domains))
        d = torch.from_numpy(np.ascontiguousarray(domains[start:stop])).to(device)
        m = torch.from_numpy(np.ascontiguousarray(masks[start:stop])).to(device)
        e = torch.from_numpy(np.ascontiguousarray(esm2[start:stop])).to(device) if use_esm else None
        batch, _ = model(d, m, e)
        logits[start:stop] = batch.float().cpu().numpy()
    return logits


@torch.inference_mode()
def score_projection(model, domains, masks, esm2, device: str, batch_size: int) -> np.ndarray:
    """The 256-dimensional contrastive embedding of every enzyme, in archive order."""
    chunks = []
    for start in range(0, len(domains), batch_size):
        stop = min(start + batch_size, len(domains))
        d = torch.from_numpy(np.ascontiguousarray(domains[start:stop])).to(device)
        m = torch.from_numpy(np.ascontiguousarray(masks[start:stop])).to(device)
        e = torch.from_numpy(np.ascontiguousarray(esm2[start:stop])).to(device)
        chunks.append(model(d, m, e).float().cpu().numpy())
    return np.concatenate(chunks)


def call_ec(logits: np.ndarray) -> np.ndarray:
    """EC calls at ``sigmoid >= 0.5``, then apply the refinement.

    Ranking uses the logits rather than their sigmoids because ``sigmoid`` saturates to 1.0 in
    float32 above a logit of roughly 17, which would turn distinct scores into ties. The two
    orderings are otherwise identical, sigmoid being strictly increasing.
    """
    scores = torch.from_numpy(logits)
    calls = scores >= 0.0
    over_cap = calls.sum(dim=1) > 5

    uncalled = ~calls.any(dim=1)
    if uncalled.any():
        best = scores[uncalled].max(dim=1, keepdim=True).values
        calls[uncalled] = scores[uncalled] >= best
    if bool(over_cap.any()):
        capped = scores[over_cap]
        cut = capped.topk(5, dim=1).values[:, -1:]
        calls[over_cap] = capped >= cut
    return calls.numpy()


def sigmoid_confidence(logits: np.ndarray, calls: np.ndarray) -> list[np.ndarray]:
    """``sigmoid(logit)`` for the called ECs of each enzyme, in ascending class order.

    Boolean masking reads the calls out in class order, which is the order ``write_predictions``
    lists them in. Only the handful of called logits are converted, so the full ``(n, 5242)``
    matrix is never duplicated, and ``torch.sigmoid`` keeps a strongly negative logit -- possible
    when an enzyme called nothing and kept its single best class -- from overflowing ``exp``.
    """
    return [torch.sigmoid(torch.from_numpy(row[call])).numpy() for row, call in zip(logits, calls, strict=True)]


def maximum_separation(distances: np.ndarray) -> int:
    """CLEAN's cut point: the first large gap in the sorted distances to the ten nearest centroids."""
    values = np.asarray(distances, dtype=np.float64)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError("maximum separation requires exactly ten finite distances")
    if np.any(values[1:] < values[:-1]):
        raise ValueError("maximum separation distances must be sorted")
    gamma = np.concatenate((values[1:], np.repeat(values[-1], 10)))
    separation = np.abs(values - gamma.mean())
    gradients = np.abs(np.diff(separation))
    large_gradients = np.flatnonzero(gradients > gradients.mean())
    chosen = int(large_gradients[0]) if len(large_gradients) else 0
    return 0 if chosen >= 5 else chosen


def apply_max_separation(
    projections: np.ndarray, centroids: np.ndarray, chunk_size: int = 512
) -> tuple[np.ndarray, list[np.ndarray]]:
    """CLEAN maximum-separation EC calls, one to five per enzyme, and the distance behind each.

    The chosen classes are sorted by index rather than kept in nearest-first order so that the
    distances line up with the way ``write_predictions`` reads the calls out of the boolean matrix.
    Sorting cannot change the calls themselves, which are a set.
    """
    queries = np.asarray(projections, dtype=np.float32)
    centers = torch.from_numpy(np.asarray(centroids, dtype=np.float32))
    calls = np.zeros((len(queries), len(centroids)), dtype=bool)
    chosen_distances: list[np.ndarray] = []
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        distances = torch.cdist(torch.from_numpy(queries[start:stop]), centers).numpy()
        for row, row_distances in enumerate(distances):
            nearest = np.argsort(row_distances, kind="stable")[:10]
            boundary = maximum_separation(row_distances[nearest])
            chosen = np.sort(nearest[: boundary + 1])
            calls[start + row, chosen] = True
            chosen_distances.append(row_distances[chosen])
    return calls, chosen_distances


def write_predictions(
    path: Path,
    entries: list[str],
    classes: np.ndarray,
    calls: np.ndarray,
    scores: list[np.ndarray] | None = None,
) -> None:
    """``Entry, predicted_ec, n_predicted``, plus ``predicted_score`` when ``scores`` is given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["Entry", "predicted_ec", "n_predicted"]
    if scores is not None:
        header.append("predicted_score")
    with path.open("w", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for position, (entry, row) in enumerate(zip(entries, calls, strict=True)):
            called = classes[row]
            record = [entry, ";".join(called), len(called)]
            if scores is not None:
                record.append(";".join(f"{value:.4f}" for value in scores[position]))
            writer.writerow(record)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npzdata", required=True, help="feature archive from process.py")
    parser.add_argument("--model", required=True, choices=sorted(MODELS), help="which released model to run")
    parser.add_argument("--checkpoint", required=True, help="weights for --model")
    parser.add_argument("--centroids", help="EC centroids for milcl_plus (default: beside its checkpoint)")
    parser.add_argument("--classes", default=str(PACKAGE_ROOT / "checkpoints" / "classes.npy"), help="EC vocabulary")
    parser.add_argument("--output", help="output CSV (default: ../outputs/<archive>_<model>.csv)")
    parser.add_argument(
        "--confscore", action="store_true", help="sigmoid probability, or centroid distance for milcl_plus"
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    spec = MODELS[args.model]
    npz_path = Path(args.npzdata).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    for path in (npz_path, checkpoint):
        if not path.is_file():
            raise SystemExit(f"no such file: {path}")
    classes = np.load(Path(args.classes).expanduser().resolve(), allow_pickle=False).astype(str)
    if len(classes) != NUM_CLASSES:
        raise SystemExit(f"expected {NUM_CLASSES} EC classes, found {len(classes)}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least one")
    destination = (
        Path(args.output).expanduser().resolve()
        if args.output
        else PACKAGE_ROOT / "outputs" / f"{npz_path.stem}_{args.model}.csv"
    )

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] {device} requested but CUDA is unavailable; using CPU")
        device = "cpu"

    entries, domains, masks, esm2 = load_features(npz_path)
    if spec["family"] == "milbc":
        # These checkpoints predate the seventeenth whole-protein token: drop it.
        model = load_milbc(str(checkpoint), spec["esm"], device)
        logits = score_sigmoid(model, domains[:, :-1, :], masks[:, :-1], esm2, spec["esm"], device, args.batch_size)
        calls = call_ec(logits)
        scores = sigmoid_confidence(logits, calls) if args.confscore else None
    else:
        centroids = (
            Path(args.centroids).expanduser().resolve() if args.centroids else checkpoint.parent / "centroids.npy"
        )
        if not centroids.is_file():
            raise SystemExit(f"milcl_plus needs EC centroids; {centroids} not found (pass --centroids)")
        model, centers = load_milcl(str(checkpoint), str(centroids), device)
        projections = score_projection(model, domains, masks, esm2, device, args.batch_size)
        calls, distances = apply_max_separation(projections, centers)
        scores = distances if args.confscore else None

    write_predictions(destination, entries, classes, calls, scores)
    summary = {
        "model": args.model,
        "rule": "sigmoid 0.5" if spec["rule"] == "sigmoid" else "CLEAN max-separation",
    }
    if args.confscore:
        summary["predicted_score"] = (
            "sigmoid probability, higher is more confident"
            if spec["rule"] == "sigmoid"
            else "distance to the EC centroid, lower is more confident"
        )
    summary.update(
        {
            "checkpoint": str(checkpoint),
            "enzymes": len(entries),
            "classes": len(classes),
            "output": str(destination),
        }
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
