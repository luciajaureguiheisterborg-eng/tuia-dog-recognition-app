"""Preprocesamiento compartido entre las notebooks y los servicios de la app."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from lib.config import settings

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAD_COLOR = tuple(int(m * 255) for m in IMAGENET_MEAN)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Convierte una imagen de OpenCV (BGR o gris) a PIL RGB."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class PadToSquare:
    """Agrega bordes para que la imagen sea cuadrada, sin deformarla."""

    def __init__(self, fill: tuple[int, int, int] = PAD_COLOR) -> None:
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        lado = max(w, h)
        lienzo = Image.new("RGB", (lado, lado), self.fill)
        lienzo.paste(img, ((lado - w) // 2, (lado - h) // 2))
        return lienzo


def eval_transform(size: int | None = None) -> T.Compose:
    """Transformación para valid, test, conjunto independiente y la app."""
    size = int(size or settings.image_size)
    return T.Compose([
        PadToSquare(),
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def preprocess_bgr(image: np.ndarray) -> torch.Tensor:
    """Imagen BGR de OpenCV -> tensor (1, 3, H, W) listo para el modelo."""
    return eval_transform()(bgr_to_pil(image)).unsqueeze(0)