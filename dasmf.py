from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class DepthAzimuthScaleRouter(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.kernel_sizes = (3, 5, 7)
        self.depth_branches = nn.ModuleList(
            nn.Conv2d(channels, channels, (k, 1), padding=(k // 2, 0),
                      groups=channels, bias=False)
            for k in self.kernel_sizes
        )
        self.azimuth_branches = nn.ModuleList(
            nn.Conv2d(channels, channels, (1, k), padding=0,
                      groups=channels, bias=False)
            for k in self.kernel_sizes
        )
        hidden = max(16, channels // 8)
        self.gate = nn.Sequential(
            nn.Conv2d(channels + 2, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 6, 1),
        )
        self.project = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        azimuth_change = (torch.roll(x, -1, dims=-1) - x).abs().mean(1, keepdim=True)
        depth_change = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(1, keepdim=True)
        depth_change = F.pad(depth_change, (0, 0, 0, 1))
        weights = self.gate(torch.cat((x, depth_change, azimuth_change), dim=1)).softmax(1)
        mixed = torch.zeros_like(x)
        for index, (kernel, branch) in enumerate(zip(self.kernel_sizes, self.depth_branches)):
            mixed = mixed + weights[:, index:index + 1] * branch(x)
        for index, (kernel, branch) in enumerate(zip(self.kernel_sizes, self.azimuth_branches)):
            wrapped = F.pad(x, (kernel // 2, kernel // 2, 0, 0), mode="circular")
            mixed = mixed + weights[:, index + 3:index + 4] * branch(wrapped)
        return x + self.project(mixed)


class MultiLevelClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleList(nn.Linear(channels, 128)
                                         for channels in (128, 256, 512))
        self.level_gate = nn.Linear(3 * 128, 3)
        self.classifier = nn.Linear(128, 4)

    def forward(self, maps):
        vectors = [F.relu(proj(F.adaptive_avg_pool2d(x, 1).flatten(1)))
                   for proj, x in zip(self.projections, maps)]
        weights = self.level_gate(torch.cat(vectors, dim=1)).softmax(dim=1)
        fused = sum(weights[:, i:i + 1] * vectors[i] for i in range(3))
        return self.classifier(fused)


class DASMFResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        grayscale_weights = backbone.conv1.weight.detach().mean(1, keepdim=True)
        backbone.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight.copy_(grayscale_weights)
        backbone.fc = nn.Linear(512, 4)
        self.backbone = backbone
        self.router = DepthAzimuthScaleRouter(128)
        self.fusion = MultiLevelClassifier()

    def forward(self, x):
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x = b.layer1(x)
        f2 = self.router(b.layer2(x))
        f3 = b.layer3(f2)
        f4 = b.layer4(f3)
        return self.fusion((f2, f3, f4))


class FocalOrdinalLoss(nn.Module):
    """Focal gamma 1.5 plus optional categorical-rank penalty (not a physical scale)."""

    def __init__(self, gamma: float = 1.5, ordinal_lambda: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.ordinal_lambda = ordinal_lambda

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        log_pt = log_probs.gather(1, targets[:, None]).squeeze(1)
        pt = log_pt.exp()
        loss = -((1 - pt) ** self.gamma) * log_pt
        if self.ordinal_lambda:
            ranks = torch.arange(logits.size(1), device=logits.device)
            rank_distance = (ranks[None, :] - targets[:, None]).abs() / 3.0
            loss = loss + self.ordinal_lambda * (probs * rank_distance).sum(1)
        return loss.mean()

def make_criterion(loss_name: str = "ce", gamma: float = 1.5,
                   ordinal_lambda: float = 0.1) -> nn.Module:
    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    if loss_name == "focal":
        return FocalOrdinalLoss(gamma=gamma, ordinal_lambda=0.0)
    if loss_name == "focal_ordinal":
        return FocalOrdinalLoss(gamma=gamma, ordinal_lambda=ordinal_lambda)
    raise ValueError(f"Unknown loss_name: {loss_name!r}")

