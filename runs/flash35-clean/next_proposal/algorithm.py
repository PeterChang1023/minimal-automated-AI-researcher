"""
Self-supervised learning algorithm for CIFAR-100 under severe compute constraints:
Decoupled Contrastive Learning with Hard Negative Weighting (DCLW) loss with a 
streamlined 2-layer projection head and optimized CIFAR augmentations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


class BarlowTwinsLoss(nn.Module):
    """
    Decoupled Contrastive Learning with Hard Negative Weighting (DCLW) loss.
    
    DCL removes the positive pair term from the InfoNCE denominator, and DCLW
    weights negative pairs proportional to their squared Euclidean distance 
    w_ij = 2 - 2 * (z_i^T z_j). This downweights false negative pairs (same class 
    images with high similarity) and emphasizes true hard negatives, leading to 
    better semantic representation alignment in few pretraining epochs.
    """

    def __init__(self, lambda_param=5e-3, embedding_dim=2048, temperature=0.1):
        super().__init__()
        self.lambda_param = lambda_param
        self.embedding_dim = embedding_dim
        self.temperature = temperature

    def forward(self, z1, z2):
        # L2 normalize projection vectors
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        batch_size = z1.size(0)
        # Concatenate features from both views: [2 * batch_size, dim]
        z = torch.cat([z1, z2], dim=0)

        # Raw cosine similarity matrix in [-1, 1]: [2N, 2N]
        raw_sim = torch.matmul(z, z.T)

        # Cosine similarity matrix scaled by temperature: [2N, 2N]
        sim = raw_sim / self.temperature

        # Target indices for positive pairs
        pos_idx = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ], dim=0)

        # Extract positive similarities: shape [2N]
        pos_sim = sim[torch.arange(2 * batch_size, device=z.device), pos_idx]

        # Mask out self-contrast (diagonal) and positive pairs from negative set
        neg_mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=z.device)
        neg_mask.fill_diagonal_(False)
        neg_mask[torch.arange(2 * batch_size, device=z.device), pos_idx] = False

        # Compute DCLW weights w_ij = 2.0 - 2.0 * raw_sim = ||z_i - z_j||^2
        w_ij = torch.clamp(2.0 - 2.0 * raw_sim, min=1e-7)

        # Add log(w_ij) in log-space for numerical stability: log(w_ij * exp(sim)) = sim + log(w_ij)
        sim_weighted = sim + torch.log(w_ij)

        # Mask invalid negative positions with a large negative number
        sim_neg = sim_weighted.masked_fill(~neg_mask, -1e9)

        # DCLW loss: - pos_sim + log( sum_{neg} w_ij * exp(sim_neg) )
        loss = -pos_sim + torch.logsumexp(sim_neg, dim=1)

        return loss.mean()


class ProjectionHead(nn.Module):
    """
    Standard 2-layer projection head (Linear -> BN -> ReLU -> Linear)
    tailored for contrastive representation learning.
    """

    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class BarlowTwinsModel(nn.Module):
    """ResNet-18 backbone adapted for CIFAR-100 + 2-layer projection head."""

    def __init__(self, proj_hidden_dim=2048, proj_output_dim=2048):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.backbone_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()

        # Adapt for CIFAR-100 (32x32): 3x3 conv1, no maxpool
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        self.backbone = backbone
        self.projector = ProjectionHead(self.backbone_dim, proj_hidden_dim, proj_output_dim)

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return features, projections


def get_model():
    """Return model with backbone and 2-layer projection head."""
    return BarlowTwinsModel(proj_hidden_dim=2048, proj_output_dim=2048)


def get_training_config():
    """Return pretraining hyperparameters tuned for contrastive learning with SGD."""
    return {
        'pretrain_epochs': 50,
        'linear_eval_epochs': 100,
        'batch_size': 512,
        'lr': 0.5,
        'weight_decay': 1e-4,
        'lambda_param': 5e-3,
        'linear_lr': 0.1,
        'linear_batch_size': 256,
    }


def get_augmentation():
    """
    Return augmentation transforms tailored for CIFAR-100 32x32 images.
    Gaussian blur is omitted as it destroys key spatial information at 32x32.
    """
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
