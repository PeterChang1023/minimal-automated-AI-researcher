"""
Self-supervised learning algorithm: InfoNCE + Feature Decorrelation Regularization on CIFAR-100.

Under severe compute constraints (130 pretrain epochs), InfoNCE contrastive learning
provides strong instance discrimination, but risks dimensional collapse where projection
features become mutually redundant. Adding a Barlow Twins-style off-diagonal cross-correlation
decorrelation penalty forces embedding dimensions to remain orthogonal, maximizing feature
capacity and accelerating representation learning.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class BarlowTwinsLoss(nn.Module):
    """InfoNCE contrastive loss combined with off-diagonal feature decorrelation regularization.

    Keeps the required class name and constructor interface for harness compatibility.
    """

    def __init__(self, lambda_param=5e-3, embedding_dim=2048, temperature=0.15):
        super().__init__()
        self.lambda_param = lambda_param
        self.embedding_dim = embedding_dim
        self.temperature = temperature

    def forward(self, z1, z2):
        """Compute combined InfoNCE + feature decorrelation loss over paired augmented views z1 and z2."""
        batch_size = z1.shape[0]

        # 1. InfoNCE (NT-Xent) Loss with L2-normalized embeddings
        z1_norm = nn.functional.normalize(z1, dim=1)
        z2_norm = nn.functional.normalize(z2, dim=1)

        out = torch.cat([z1_norm, z2_norm], dim=0)
        similarity_matrix = torch.mm(out, out.t()) / self.temperature
        similarity_matrix.fill_diagonal_(-1e9)

        targets = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=z1.device),
                torch.arange(0, batch_size, device=z1.device),
            ],
            dim=0,
        )
        loss_infonce = nn.functional.cross_entropy(similarity_matrix, targets)

        # 2. Off-diagonal Cross-Correlation Decorrelation Regularization
        z1_bn = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-5)
        z2_bn = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-5)

        c = torch.mm(z1_bn.T, z2_bn) / batch_size
        off_diag_loss = c.pow(2).sum() - torch.diagonal(c).pow(2).sum()

        return loss_infonce + self.lambda_param * off_diag_loss


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
