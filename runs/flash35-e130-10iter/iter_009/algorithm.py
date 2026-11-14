"""
Self-supervised learning algorithm: InfoNCE (SimCLR) contrastive learning on CIFAR-100.

Under severe compute constraints (130 pretrain epochs), contrastive loss (InfoNCE)
provides stronger instance-discrimination gradients via negative pair repulsion
than cross-correlation decorrelation (Barlow Twins), leading to faster convergence
and better representation quality under short training schedules.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class BarlowTwinsLoss(nn.Module):
    """InfoNCE (NT-Xent) loss replacing cross-correlation loss for fast SSL convergence.

    Keeps the required class name and constructor interface for harness compatibility.
    """

    def __init__(self, lambda_param=5e-3, embedding_dim=2048, temperature=0.15):
        super().__init__()
        self.lambda_param = lambda_param
        self.embedding_dim = embedding_dim
        self.temperature = temperature

    def forward(self, z1, z2):
        """Compute InfoNCE contrastive loss over paired augmented views z1 and z2."""
        batch_size = z1.shape[0]

        # L2-normalize projection vectors
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        # Concatenate embeddings: [2B, D]
        out = torch.cat([z1, z2], dim=0)

        # Cosine similarity matrix scaled by temperature: [2B, 2B]
        similarity_matrix = torch.mm(out, out.t()) / self.temperature

        # Mask out self-similarity on the diagonal
        similarity_matrix.fill_diagonal_(-1e9)

        # Targets: z1_i matches z2_i (index i + B), z2_i matches z1_i (index i)
        targets = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=z1.device),
                torch.arange(0, batch_size, device=z1.device),
            ],
            dim=0,
        )

        return nn.functional.cross_entropy(similarity_matrix, targets)


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
        transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
