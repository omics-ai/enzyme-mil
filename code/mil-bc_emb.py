import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True, help='Path to config.yaml')
args = parser.parse_args()
config = yaml.safe_load(open(args.config_path, 'r'))


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs


def load_dataset_labels(npz_path='split100_esm.npz'):
    dataset_samples = np.load(npz_path, allow_pickle=True)['dataset_samples']
    labels = []
    for sample in dataset_samples:
        label_mlb = sample['label_mlb']
        labels.append(label_mlb)
    y_numpy = np.stack(labels, axis=0)  # Shape: (num_samples, 1, 5242)
    y_numpy = y_numpy.squeeze(axis=1)  # Reshape to (num_samples, 5242)
    return y_numpy


def compute_class_weights(npz_path, min_weight=1.0, max_weight=1000.0):
    labels = load_dataset_labels(npz_path)  # Shape: (227360, 5242)

    pos_counts = labels.sum(axis=0)  # Count positive samples for each label
    neg_counts = len(labels) - pos_counts
    class_weights = neg_counts / (pos_counts + 1e-8)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    clamped_weights = torch.clamp(class_weights, min=min_weight, max=max_weight)
    clamped_smoothed_weights = torch.log1p(clamped_weights)

    return clamped_smoothed_weights


class ENZDataset(torch.utils.data.Dataset):
    def __init__(self, source='split100'):
        npz_path = os.path.join(config['project']['data_dir'], f'{source}_esm_17dct.npz')
        dataset_samples = np.load(npz_path, allow_pickle=True)['dataset_samples']
        self.instance_feature = []
        self.bag_feature = []
        self.masks = []
        self.labels = []
        for sample in dataset_samples:
            # entry = sample['entry']
            # dct_count = sample['dct_count']
            # dct_feature = sample['dct_feature'][:, :480]  # (16,480)
            dct_instance = sample['dct_feature']  # (17,480)
            # dct_global = sample['dct_global']  # (480,)
            dct_mask = sample['dct_mask']
            # label_ec = sample['label_ec']
            label_mlb = sample['label_mlb']
            esm2_feature = sample['esm2_feature']
            self.instance_feature.append(dct_instance)
            self.bag_feature.append(esm2_feature)
            self.masks.append(dct_mask)
            self.labels.append(label_mlb)
        print(npz_path, len(self.instance_feature), len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = torch.tensor(self.instance_feature[index], dtype=torch.float32)
        mask = torch.tensor(self.masks[index], dtype=torch.bool)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return sample, mask, label


class AttentionMIL(nn.Module):
    def __init__(self, num_classes, instance_input_dim=480, instance_embedding_dim=512,
                 attention_hidden_dim=128, classifier_hidden_dim=128, dropout_rate=0.25):
        """
        Args:
            num_classes (int): Number of output classes.
            instance_input_dim (int): Dimension of the input instance embedding (960 in your case).
            instance_embedding_dim (int): Dimension of each instance embedding.
            attention_hidden_dim (int): Hidden dimension for the attention mechanism.
            classifier_hidden_dim (int): Hidden dimension for the classifier.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(AttentionMIL, self).__init__()

        # Instance-level Feature Transformation
        self.instance_encoder = nn.Sequential(
            nn.Linear(instance_input_dim, instance_embedding_dim),
            nn.LayerNorm(instance_embedding_dim),
            nn.ReLU()
        )

        # Attention Mechanism
        #  It follows the Gated Attention mechanism structure: e_k = w_att^T * tanh(V_att * h_k)
        #  V_att: maps h_k from embedding_dim to attention_hidden_dim
        self.attention_V = nn.Sequential(
            nn.Linear(instance_embedding_dim, attention_hidden_dim),
            nn.LayerNorm(attention_hidden_dim),
            nn.Tanh()
        )
        self.attention_gate = nn.Sequential(
            nn.Linear(instance_embedding_dim, attention_hidden_dim),
            nn.LayerNorm(attention_hidden_dim),
            nn.Sigmoid()
        )
        # w_att: maps u_k from attention_hidden_dim to a single attention score
        self.attention_w = nn.Linear(attention_hidden_dim, 1)

        # Classification Head
        #  bag-level representation used for classification is a weighted sum of the instance embeddings,
        #  and those embeddings are of dimension self.instance_embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(instance_embedding_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(classifier_hidden_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def forward(self, instances_in_bag, mask=None):
        """
        Forward pass of the model.

        Args:
            instances_in_bag (torch.Tensor): representing a batch of bags.
                                          shape (batch_size, num_instances, instance_embedding_dim)
                                          num_instances should be max_instances (e.g., 16).
            mask (torch.Tensor, optional): shape (batch_size, num_instances)
                                           with 1 for real instances and 0 for padded instances.

        Returns:
            logits (torch.Tensor): Raw output scores from the classifier (batch_size, num_classes).
            attention_weights (torch.Tensor): Attention weights for instances (batch_size, num_instances, 1).
        """
        batch_size, num_instances, _ = instances_in_bag.shape

        # Step 1: Instance-level Feature Transformation
        #  instances_in_bag shape: (batch_size, num_instances, instance_input_dim)
        #  h_k shape: (batch_size, num_instances, instance_embedding_dim)
        h_k = self.instance_encoder(instances_in_bag)

        # Step 2: Attention Mechanism
        #  Calculate u_k = tanh(V_att * h_k)
        #  self.attention_V(h_k) results in shape: (batch_size, num_instances, attention_hidden_dim)
        u_k = self.attention_V(h_k)
        g_k = self.attention_gate(h_k)
        u_k = u_k * g_k  # Element-wise multiplication

        # Calculate e_k = w_att^T * u_k (unnormalized attention scores)
        #  self.attention_w(u_k) results in shape: (batch_size, num_instances, 1)
        e_k = self.attention_w(u_k).squeeze(-1)  # Shape: (batch_size, num_instances)

        # Apply mask before softmax: set scores of padded segments to a very small number
        if mask is not None:
            # masked_fill_ expects a boolean mask (True where condition is met)
            # We want to fill where mask is 0 (padded segment)
            e_k.masked_fill_(mask == 0, -1e9)

        # Calculate attention weights alpha_k = softmax(e_k)
        # Softmax is applied across the segments (dim=1)
        alpha_k = nn.functional.softmax(e_k, dim=1)  # Shape: (batch_size, num_instances)

        # Step 3: Bag Representation Generation
        #  S_bag = sum_k (alpha_k * h_k)
        #  alpha_k needs to be (B, K_max, 1) to multiply with h_k (B, K_max, D_hidden)
        #  The result is then summed over the K_max dimension.
        bag_representation = torch.sum(alpha_k.unsqueeze(-1) * h_k, dim=1)  # Shape: (batch_size, embedding_dim)

        # Step 4: Classification Head
        #  logits shape: (batch_size, num_labels)
        logits = self.classifier(bag_representation)

        return logits, bag_representation


def train_model(num_total_labels=5242, instance_input_dim=480,
                model_embedding_dim=512, model_attention_hidden_dim=128,
                model_classifier_hidden_dim=128, model_dropout_rate=0.25,
                num_epochs=100, batch_size=64, learning_rate=0.0002, weight_decay=0.0001):
    # Load dataset
    dataset_train = ENZDataset(source='split100')
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Initialize the model
    model = AttentionMIL(num_classes=num_total_labels,
                         instance_input_dim=instance_input_dim,
                         instance_embedding_dim=model_embedding_dim,
                         attention_hidden_dim=model_attention_hidden_dim,
                         classifier_hidden_dim=model_classifier_hidden_dim,
                         dropout_rate=model_dropout_rate
                         )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    npz_path = os.path.join(config['project']['data_dir'], 'split100_esm.npz')
    class_weight = compute_class_weights(npz_path, min_weight=1.0, max_weight=1000.0)
    class_weight = class_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=5)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs - 5)

    model_dir = config['project']['model_dir']
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    name = 'milBC'
    model_name = '{}-{}-{}-{}-r{}-l{}-w{}-b{}'.format(name,
                                                      model_embedding_dim,
                                                      model_attention_hidden_dim,
                                                      model_classifier_hidden_dim,
                                                      model_dropout_rate,
                                                      learning_rate,
                                                      weight_decay,
                                                      batch_size
                                                      )

    for epoch in range(num_epochs):
        model_save_path = os.path.join(model_dir, model_name + '_{:03d}.pth'.format(epoch))

        model.train()
        training_loss = 0.0
        true_labels, predict_labels = [], []
        for batch_idx, (samples, masks, labels) in enumerate(train_loader):
            samples = samples.to(device)
            masks = masks.to(device)
            labels = labels.to(device).squeeze(1)

            optimizer.zero_grad()
            logits, _ = model(samples, masks)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            training_loss += loss.item() * labels.size(0)
            prediction_prob = torch.sigmoid(logits)
            prediction = (prediction_prob > 0.5).float()
            true_labels.extend(labels.cpu().numpy())
            predict_labels.extend(prediction.cpu().numpy())

        if epoch < 5:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        torch.save(model.state_dict(), model_save_path)


seed_everything(42)
train_model(model_embedding_dim=config['model']['instance_embedding_dim'],
            model_attention_hidden_dim=config['model']['attention_hidden_dim'],
            model_classifier_hidden_dim=config['model']['classifier_hidden_dim'],
            model_dropout_rate=config['model']['dropout_rate'],
            num_epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'])
