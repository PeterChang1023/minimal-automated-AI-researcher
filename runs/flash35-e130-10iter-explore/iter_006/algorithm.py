"""
Self-supervised learning algorithm: Decoupled Contrastive Learning (DCL)
with CIFAR-adapted ResNet-34 Backbone, Output-BN Projection Head,
and Solarization Augmentation.

Combines deeper hierarchical representations from ResNet-34 (16 basic blocks),
output-standardized projection heads, sharper DCL temperature (tau=0.08),
and solarization data augmentation for fast, high-quality feature learning on CIFAR-100.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class BarlowTwinsLoss(nn.Module):
    """
    Decoupled Contrastive Loss (DCL) implemented inside the BarlowTwinsLoss wrapper.
    
    DCL removes positive similarity from the loss denominator, preventing high positive
    alignment from dampening its own gradient during optimization.
    """

    def __init__(self, lambda_param=5e-3, embedding_dim=2048, temperature=0.08):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size = z1.shape[0]

        # L2-normalize projection vectors
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)

        # Concatenate projections: shape [2 * batch_size, embedding_dim]
        z = torch.cat([z1_norm, z2_norm], dim=0)

        # Cosine similarity matrix scaled by temperature
        sim = torch.matmul(z, z.T) / self.temperature

        # Extract positive pair similarities
        pos_sim = torch.cat([
            torch.diag(sim, batch_size),
            torch.diag(sim, -batch_size)
        ], dim=0)  # shape [2 * batch_size]

        # Mask out self-similarity (diagonal) and positive pair similarity
        mask_self = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        mask_pos = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=z.device)
        mask_pos[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        mask_pos[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool, device=z.device)

        mask_neg = ~(mask_self | mask_pos)

        # Mask non-negative entries with large negative value for logsumexp
        sim_neg = sim.masked_fill(~mask_neg, -1e9)

        # Decoupled contrastive loss: - pos_sim + logsumexp(neg_sim)
        neg_logsumexp = torch.logsumexp(sim_neg, dim=1)
        loss = (-pos_sim + neg_logsumexp).mean()

        return loss


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFARResNet34(nn.Module):
    """CIFAR-adapted ResNet-34 backbone (layers [3, 4, 6, 3], channels [64, 128, 256, 512])."""

    def __init__(self):
        super().__init__()
        self.in_planes = 64

        # CIFAR stem: 3x3 conv, stride 1, no maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet-34 layer block configuration: [3, 4, 6, 3]
        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


class ProjectionHead(nn.Module):
    """3-layer projection head with BatchNorm on all layers including output."""

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
            nn.BatchNorm1d(output_dim, affine=False),
        )

    def forward(self, x):
        return self.net(x)


class BarlowTwinsModel(nn.Module):
    """CIFAR ResNet-34 backbone + BatchNorm projection head."""

    def __init__(self, proj_hidden_dim=2048, proj_output_dim=2048):
        super().__init__()
        self.backbone = CIFARResNet34()
        self.backbone_dim = 512
        self.projector = ProjectionHead(self.backbone_dim, proj_hidden_dim, proj_output_dim)

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return features, projections


def get_model():
    """Return model instance with ResNet-34 backbone and projection head."""
    return BarlowTwinsModel(proj_hidden_dim=2048, proj_output_dim=2048)


def get_training_config():
    """Return training hyperparameters."""
    return {
        'pretrain_epochs': 130,
        'linear_eval_epochs': 100,
        'batch_size': 512,
        'lr': 0.35,
        'weight_decay': 1e-4,
        'lambda_param': 5e-3,
        'linear_lr': 0.1,
        'linear_batch_size': 256,
    }


def get_augmentation():
    """Return CIFAR-100 augmentation pipeline for DCL contrastive pretraining."""
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
        transforms.RandomSolarize(threshold=128, p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
