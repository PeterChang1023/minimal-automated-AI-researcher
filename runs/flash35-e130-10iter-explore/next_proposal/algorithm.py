"""
Self-supervised learning algorithm: Decoupled Contrastive Learning (DCL)
with Anti-Aliased CIFAR-ResNet-26 Backbone and Multi-Stage Feature Aggregation.

Aggregates L2-normalized stage-3 (384-d) and stage-4 (768-d) representations into
an 1152-dimensional multi-scale backbone feature vector, paired with a wider 3-layer MLP
projection head (1152 -> 4096 -> 4096 -> 2048) and BlurPool anti-aliased downsampling.
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

    def __init__(self, lambda_param=5e-3, embedding_dim=2048, temperature=0.10):
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


class BlurPool2d(nn.Module):
    """
    Anti-aliased downsampling (BlurPool) module using a 3x3 binomial filter
    [1, 2, 1] x [1, 2, 1]^T / 16 (Zhang 2019).
    """

    def __init__(self, channels, stride=2):
        super().__init__()
        self.stride = stride
        kernel = torch.tensor([
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0]
        ])
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=self.stride, padding=1, groups=x.shape[1])


class AntiAliasedBasicBlock(nn.Module):
    """ResNet BasicBlock with BlurPool anti-aliased downsampling."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        if stride == 2:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.blur1 = BlurPool2d(planes, stride=2)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.blur1 = nn.Identity()

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if stride == 2:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                    BlurPool2d(self.expansion * planes, stride=2),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.conv1(x)
        out = self.blur1(out)
        out = F.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFARResNet26AntiAliasedMultiStage(nn.Module):
    """
    CIFAR-adapted ResNet-26 ([3, 3, 3, 3] blocks) with anti-aliasing and
    multi-stage feature aggregation (combining layer3 384-d and layer4 768-d features -> 1152-d).
    """

    def __init__(self, width_mult=1.5):
        super().__init__()
        base_channels = [int(c * width_mult) for c in [64, 128, 256, 512]]  # [96, 192, 384, 768]
        self.in_planes = base_channels[0]

        # CIFAR stem: 3x3 conv, stride 1, no maxpool
        self.conv1 = nn.Conv2d(3, base_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels[0])

        # Layer depths: [3, 3, 3, 3] = ResNet-26
        self.layer1 = self._make_layer(AntiAliasedBasicBlock, base_channels[0], 3, stride=1)
        self.layer2 = self._make_layer(AntiAliasedBasicBlock, base_channels[1], 3, stride=2)
        self.layer3 = self._make_layer(AntiAliasedBasicBlock, base_channels[2], 3, stride=2)
        self.layer4 = self._make_layer(AntiAliasedBasicBlock, base_channels[3], 3, stride=2)
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
        out3 = self.layer3(out)
        out4 = self.layer4(out3)

        # Extract, pool, and L2-normalize stage-3 (384-d) and stage-4 (768-d) features
        feat3 = F.normalize(torch.flatten(self.avgpool(out3), 1), dim=1)
        feat4 = F.normalize(torch.flatten(self.avgpool(out4), 1), dim=1)

        # Concatenate into 1152-dimensional multi-stage representation
        features = torch.cat([feat3, feat4], dim=1)
        return features


class ProjectionHead(nn.Module):
    """Wider 3-layer projection head (1152 -> 4096 -> 4096 -> 2048)."""

    def __init__(self, input_dim=1152, hidden_dim=4096, output_dim=2048):
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
    """
    CIFAR ResNet-26 Anti-Aliased backbone with multi-stage feature aggregation (1152-d)
    and 4096-wide 3-layer projection head.
    """

    def __init__(self, proj_hidden_dim=4096, proj_output_dim=2048):
        super().__init__()
        self.backbone = CIFARResNet26AntiAliasedMultiStage(width_mult=1.5)
        self.backbone_dim = 1152
        self.projector = ProjectionHead(self.backbone_dim, proj_hidden_dim, proj_output_dim)

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return features, projections


def get_model():
    """Return model instance with multi-stage anti-aliased ResNet-26 backbone."""
    return BarlowTwinsModel(proj_hidden_dim=4096, proj_output_dim=2048)


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
    """Return CIFAR-100 detail-preserving augmentation pipeline."""
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
