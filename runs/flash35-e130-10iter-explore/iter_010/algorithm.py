"""
Self-supervised learning algorithm: Decoupled Contrastive Loss with Queue Memory Bank (DCL-Q)
with Wider CIFAR-ResNet-18 Backbone.

DCL-Q combines Decoupled Contrastive Loss (DCL) with a memory bank queue (4096 embeddings)
to expand the negative representation space from 1022 to 5118 negative pairs per sample,
enhancing representation uniformity and class separation on CIFAR-100.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class BarlowTwinsLoss(nn.Module):
    """
    Decoupled Contrastive Loss with Queue Memory Bank (DCL-Q).
    
    DCL removes positive similarity from the denominator to prevent gradient dampening,
    while the queue buffer expands negative pair coverage across recent mini-batches.
    """

    def __init__(self, lambda_param=5e-3, embedding_dim=2048, temperature=0.10, queue_size=4096):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        self.embedding_dim = embedding_dim

        # Register queue memory buffer and queue pointer
        queue = F.normalize(torch.randn(queue_size, embedding_dim), dim=1)
        self.register_buffer("queue", queue)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Enqueue new projection keys and dequeue oldest keys in FIFO fashion."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())

        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
            ptr = (ptr + batch_size) % self.queue_size
        else:
            overflow = (ptr + batch_size) - self.queue_size
            self.queue[ptr:] = keys[:self.queue_size - ptr]
            self.queue[:overflow] = keys[self.queue_size - ptr:]
            ptr = overflow

        self.queue_ptr[0] = ptr

    def forward(self, z1, z2):
        batch_size = z1.shape[0]

        # L2-normalize projection vectors
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)

        # Concatenate current batch projections: shape [2 * batch_size, embedding_dim]
        z = torch.cat([z1_norm, z2_norm], dim=0)

        # 1. Cosine similarity matrix for current batch scaled by temperature
        sim_batch = torch.matmul(z, z.T) / self.temperature

        # Extract positive pair similarities (z1_i with z2_i)
        pos_sim = torch.cat([
            torch.diag(sim_batch, batch_size),
            torch.diag(sim_batch, -batch_size)
        ], dim=0)  # shape [2 * batch_size]

        # Mask out self-similarity (diagonal) and positive pair similarity
        mask_self = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        mask_pos = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=z.device)
        mask_pos[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        mask_pos[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool, device=z.device)

        mask_neg = ~(mask_self | mask_pos)
        sim_batch_neg = sim_batch.masked_fill(~mask_neg, -1e9)

        # 2. Similarity with queue negative keys: shape [2 * batch_size, queue_size]
        sim_queue = torch.matmul(z, self.queue.clone().detach().T) / self.temperature

        # Combine in-batch negative similarities and queue negative similarities
        sim_all_neg = torch.cat([sim_batch_neg, sim_queue], dim=1)

        # Decoupled contrastive loss: - pos_sim + logsumexp(all_negatives)
        neg_logsumexp = torch.logsumexp(sim_all_neg, dim=1)
        loss = (-pos_sim + neg_logsumexp).mean()

        # Update memory queue with current batch projections if in training mode
        if self.training:
            self._dequeue_and_enqueue(z.detach())

        return loss


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock with customizable channel width."""
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


class CIFARResNet18Wider(nn.Module):
    """CIFAR-adapted ResNet-18 with 1.5x width multiplier (channels [96, 192, 384, 768])."""

    def __init__(self, width_mult=1.5):
        super().__init__()
        base_channels = [int(c * width_mult) for c in [64, 128, 256, 512]]  # [96, 192, 384, 768]
        self.in_planes = base_channels[0]

        # CIFAR stem: 3x3 conv, stride 1, no maxpool
        self.conv1 = nn.Conv2d(3, base_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels[0])

        self.layer1 = self._make_layer(BasicBlock, base_channels[0], 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, base_channels[1], 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, base_channels[2], 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, base_channels[3], 2, stride=2)
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
    """3-layer projection head."""

    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=2048):
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
    """Wider ResNet-18 backbone + 3-layer projection head."""

    def __init__(self, proj_hidden_dim=2048, proj_output_dim=2048):
        super().__init__()
        self.backbone = CIFARResNet18Wider(width_mult=1.5)
        self.backbone_dim = 768
        self.projector = ProjectionHead(self.backbone_dim, proj_hidden_dim, proj_output_dim)

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return features, projections


def get_model():
    """Return model instance with wider backbone and projection head."""
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
    """Return CIFAR-100 augmentation pipeline for DCL contrastive pretraining."""
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
