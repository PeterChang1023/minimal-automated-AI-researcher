"""
Self-supervised learning algorithm: Decoupled Contrastive Learning (DCL) on CIFAR-100.

Under compute constraints (130 pretrain epochs), Decoupled Contrastive Loss (DCL)
removes the positive pair term from the denominator of the contrastive loss function.
This eliminates positive-negative gradient coupling and prevents negative gradient
suppression, leading to faster representation learning than standard InfoNCE.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class BarlowTwinsLoss(nn.Module):
    """Decoupled Contrastive Loss (DCL) replacing cross-correlation loss.

    Keeps the required class name and constructor interface for harness compatibility.
    """

    def __init__(self, lambda_param=5e-3, embedding_dim=2048, temperature=0.1):
        super().__init__()
        self.lambda_param = lambda_param
        self.embedding_dim = embedding_dim
        self.temperature = temperature

    def forward(self, z1, z2):
        """Compute Decoupled Contrastive Loss (DCL) over paired augmented views z1 and z2."""
        batch_size = z1.shape[0]

        # L2-normalize projection vectors
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        # Positive pair similarity (dot product divided by temperature)
        pos_sim = (z1 * z2).sum(dim=-1) / self.temperature

        # Compute cosine similarity matrices scaled by temperature
        sim_11 = torch.mm(z1, z1.t()) / self.temperature
        sim_12 = torch.mm(z1, z2.t()) / self.temperature
        sim_22 = torch.mm(z2, z2.t()) / self.temperature

        # Create diagonal mask to exclude self-similarity and positive pairs from negative pool
        mask_diag = torch.eye(batch_size, dtype=torch.bool, device=z1.device)

        sim_11 = sim_11.masked_fill(mask_diag, -1e9)
        sim_22 = sim_22.masked_fill(mask_diag, -1e9)
        sim_12 = sim_12.masked_fill(mask_diag, -1e9)
        sim_21 = sim_12.t()

        # DCL loss for z1 as anchor (negatives are all z1_k and z2_k for k != i)
        neg_1 = torch.cat([sim_11, sim_12], dim=1)
        loss_1 = -pos_sim + torch.logsumexp(neg_1, dim=1)

        # DCL loss for z2 as anchor
        neg_2 = torch.cat([sim_22, sim_21], dim=1)
        loss_2 = -pos_sim + torch.logsumexp(neg_2, dim=1)

        return (loss_1.mean() + loss_2.mean()) / 2.0


class ProjectionHead(nn.Module):
    """3-layer projection head for self-supervised pretraining."""

    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class BarlowTwinsModel(nn.Module):
    """ResNet-18 backbone + projection head adapted for CIFAR-100."""

    def __init__(self, proj_hidden_dim=2048, proj_output_dim=2048):
        super().__init__()
        # ResNet-18 backbone (remove final FC)
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
    """Return model instance."""
    return BarlowTwinsModel(proj_hidden_dim=2048, proj_output_dim=2048)


def get_training_config():
    """Return training hyperparameters."""
    return {
        'pretrain_epochs': 130,
        'linear_eval_epochs': 100,
        'batch_size': 512,
        'lr': 0.4,
        'weight_decay': 1e-4,
        'lambda_param': 5e-3,
        'linear_lr': 0.1,
        'linear_batch_size': 256,
    }


def get_augmentation():
    """Return CIFAR-100 data augmentations."""
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
