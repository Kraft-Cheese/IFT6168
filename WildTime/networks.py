"""
Network architectures for Wild-Time experiments.
Matches the architectures used in the original Wild-Time benchmark.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class YearbookCNN(nn.Module):
    """
    4-layer CNN for Yearbook (32x32 grayscale-ish images, 2 classes).
    Architecture matches Wild-Time's default.
    """

    def __init__(self, num_classes=2, num_input_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(num_input_channels, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(32)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.mean(dim=[2, 3])  # global average pooling
        return self.classifier(x)


class DenseNet121(nn.Module):
    """
    DenseNet-121 pretrained on ImageNet for FMoW (62-class satellite imagery).
    Matches Wild-Time's default architecture for FMoW.
    """

    def __init__(self, num_classes=62, pretrained=True):
        super().__init__()
        from torchvision import models
        self.net = models.densenet121(pretrained=pretrained)
        self.net.classifier = nn.Linear(self.net.classifier.in_features, num_classes)

    def forward(self, x):
        return self.net(x)


NETWORK_REGISTRY = {
    "yearbook_cnn": YearbookCNN,
    "densenet121": DenseNet121,
}


def get_network(name, **kwargs):
    if name not in NETWORK_REGISTRY:
        raise ValueError(f"Unknown network: {name}. Available: {list(NETWORK_REGISTRY.keys())}")
    return NETWORK_REGISTRY[name](**kwargs)
