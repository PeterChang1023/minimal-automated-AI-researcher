"""
Self-supervised learning algorithm: SimCLR InfoNCE Loss (tau=0.1) on CIFAR-100.

In a compute-constrained setting (50 epochs), InfoNCE contrastive learning
with a tuned temperature parameter (tau=0.1) provides stronger hard-negative
gradients, accelerating representation learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


class BarlowTwinsLoss(nn.Module):
    """InfoNCE (SimCLR) loss wrapped in BarlowTwinsLoss interface."""

    def __init__(self, lambda_param=5e-3, embedding_dim=2048, temperature=0.1):
        super().__init__()
        self.lambda_param = lambda_param
        self.embedding_dim = embedding_dim
        self.temperature = temperature

    def forward(self, z1, z2):
        N = z1.size(0)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate positive pair representations: [2N, D]
        z = torch.cat([z1, z2], dim=0)

        # Cosine similarity matrix: [2N, 2N]
        sim = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -1e4)

        # Targets: for view 1 sample i -> view 2 sample i (index i + N)
        #          for view 2 sample i -> view 1 sample i (index i)
        targets = torch.arange(N, device=z.device)
        targets = torch.cat([targets + N, targets], dim=0)

        return F.cross_entropy(sim, targets)


class ProjectionHead(nn.Module):
    """3-layer projection head."""

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
    """ResNet-18 backbone adapted for CIFAR-100 + projection head."""

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
    """Return model instance."""
    return BarlowTwinsModel(proj_hidden_dim=2048, proj_output_dim=2048)


def get_training_config():
    """Return training hyperparameters."""
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
    """Return augmentation transforms for CIFAR-100."""
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
