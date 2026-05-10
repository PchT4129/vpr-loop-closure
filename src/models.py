import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = torch.flatten(features, start_dim=1)
        features = F.normalize(features, p=2, dim=1)
        return features