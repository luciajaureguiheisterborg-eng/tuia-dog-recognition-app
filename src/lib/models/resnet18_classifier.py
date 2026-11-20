"""Modelo A: ResNet18 preentrenada en ImageNet con una cabeza nueva para las razas."""
from __future__ import annotations

import torch
from torch import nn


class ResNet18Classifier(nn.Module):
    """Backbone ResNet18 (salida de 512) + Dropout + capa lineal de clasificacion.

    El embedding es la salida del backbone (penultima capa, 512 dims), por lo que
    EMBEDDING_DIM debe ser 512 para este modelo.
    """

    EMBEDDING_DIM = 512

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.3,
        classes: list[str] | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        self.classes = classes
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.EMBEDDING_DIM, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.embed(x)))