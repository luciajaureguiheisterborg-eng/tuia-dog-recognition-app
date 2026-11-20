"""Modelo B: CNN propia entrenada desde cero.

Vive en un modulo importable (no en la notebook) para que torch.load pueda reconstruir
la clase cuando la app carga el checkpoint.
"""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Dos convoluciones 3x3 + BatchNorm + ReLU y un MaxPool que achica la imagen a la mitad."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNCustom(nn.Module):
    """5 bloques convolucionales -> GAP -> capa de embedding -> Dropout -> clasificador."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        dropout: float = 0.4,
        classes: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.classes = classes  # lista de razas: se guarda junto con el modelo
        chans = [3, 32, 64, 128, 256, 256]
        self.features = nn.Sequential(*[ConvBlock(chans[i], chans[i + 1]) for i in range(5)])
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(chans[-1], embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Vector de embedding (embedding_dim) de cada imagen del batch."""
        return self.embedding(self.pool(self.features(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.embed(x)))