from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import copy
import json
import os
import random
import time

import cv2
import numpy as np
import onnxruntime
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lib.config import settings
from lib.preprocessing import bgr_to_pil, eval_transform, train_transform

logger = logging.getLogger(__name__)

# Hiperparametros por defecto de cada modelo. Se pueden pisar desde la notebook con
#   classifier.hparams_override = {"cnn_custom": {"epochs": 3}}
DEFAULT_HPARAMS: dict[str, dict[str, Any]] = {
    "resnet18_finetuned": {
        "strategy": "fine_tuning",  # o "feature_extraction" (solo se entrena la cabeza)
        "epochs": 15,
        "patience": 4,
        "batch_size": 64,
        "lr_head": 1e-3,
        "lr_backbone": 1e-4,  # learning rate diferencial: el backbone ya sabe ver, se ajusta poco
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "label_smoothing": 0.1,
    },
    "cnn_custom": {
        "epochs": 40,
        "patience": 8,
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.4,
        "label_smoothing": 0.1,
    },
}


class BreedDataset(Dataset):
    """Lee imagenes con OpenCV (BGR, como la app) y aplica el transform indicado."""

    def __init__(self, root: Path, paths: list[str], labels: list[int], transform) -> None:
        self.root = Path(root)
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = Path(self.paths[idx])
        if not path.is_absolute() and not path.exists():
            path = self.root / path
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"No se pudo leer la imagen: {path}")
        return self.transform(bgr_to_pil(image)), self.labels[idx]


class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    # --- helpers privados de la Etapa 2 ---------------------------------

    def _hparams(self) -> dict[str, Any]:
        hp = dict(DEFAULT_HPARAMS[self.active_model_name])
        hp.update(getattr(self, "hparams_override", {}).get(self.active_model_name, {}))
        return hp

    def _load_splits(self):
        """Lee data/splits.csv (generado en 1-pre-processing.ipynb). Rutas relativas a dataset_path."""
        import pandas as pd

        splits_file = self.dataset_path.parent / "splits.csv"
        if not splits_file.exists():
            raise FileNotFoundError(
                f"No existe {splits_file}. Generalo con 1-pre-processing.ipynb (seccion de splits)."
            )
        df = pd.read_csv(splits_file)
        classes = sorted(df["breed"].unique())  # mismo orden siempre: indice <-> raza
        return df, classes

    def _make_loader(self, paths, labels, transform, batch_size, shuffle, drop_last=False):
        workers = min(4, os.cpu_count() or 1)
        return DataLoader(
            BreedDataset(self.dataset_path, list(paths), list(labels), transform),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
        )

    def _build_model(self, classes: list[str], hp: dict[str, Any]) -> nn.Module:
        if self.active_model_name == "resnet18_finetuned":
            from lib.models.resnet18_classifier import ResNet18Classifier

            return ResNet18Classifier(len(classes), dropout=hp["dropout"], classes=classes)
        if self.active_model_name == "cnn_custom":
            from lib.models.cnn_custom import CNNCustom

            return CNNCustom(
                len(classes),
                embedding_dim=int(settings.embedding_dim),
                dropout=hp["dropout"],
                classes=classes,
            )
        raise ValueError(f"Modelo no soportado: {self.active_model_name}")

    @staticmethod
    def _run_epoch(model, loader, criterion, device, optimizer=None, freeze_backbone=False):
        """Una pasada por el loader. Con optimizer entrena; sin optimizer solo evalua."""
        training = optimizer is not None
        model.train(training)
        if training and freeze_backbone:
            model.backbone.eval()  # BatchNorm del backbone congelado: no actualiza sus estadisticas
        total_loss, correct, n = 0.0, 0, 0
        with torch.set_grad_enabled(training):
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * x.size(0)
                correct += int((logits.argmax(1) == y).sum())
                n += x.size(0)
        return total_loss / n, correct / n

    # --- funciones de la Etapa 2 ----------------------------------------

    def train_classifier(self) -> None:
        """
        Entrena el clasificador de razas sobre el dataset (self.dataset_path).

        Modelo A (obligatorio): fine-tuning de ResNet18 (o ResNet50) pre-entrenado.
        Modelo B (obligatorio): CNN propia.

        Debe:
          - Usar los splits train/valid definidos en 1-pre-processing.ipynb.
          - Aplicar el preprocesamiento y data augmentation justificados.
          - Guardar el checkpoint resultante en self.active_checkpoint
            (ej: models/resnet18_finetuned.pth).
        """
        hp = self._hparams()
        name = self.active_model_name
        seed = int(hp.get("seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        df, classes = self._load_splits()
        class_to_idx = {c: i for i, c in enumerate(classes)}
        train_df, valid_df = df[df["split"] == "train"], df[df["split"] == "valid"]
        y_train = train_df["breed"].map(class_to_idx).tolist()
        y_valid = valid_df["breed"].map(class_to_idx).tolist()

        # Augmentation solo en train; valid usa el mismo transform que test y la app.
        train_loader = self._make_loader(
            train_df["path"], y_train, train_transform(self.image_size), hp["batch_size"],
            shuffle=True, drop_last=True,
        )
        valid_loader = self._make_loader(
            valid_df["path"], y_valid, eval_transform(self.image_size), hp["batch_size"],
            shuffle=False,
        )

        # Desbalance de clases (65-198 imagenes por raza): peso inversamente proporcional a la frecuencia.
        counts = np.bincount(y_train, minlength=len(classes)).astype(np.float64)
        weights = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=hp["label_smoothing"])
        # En valid se mide sin pesos ni suavizado, para que la loss sea comparable entre experimentos.
        valid_criterion = nn.CrossEntropyLoss()

        model = self._build_model(classes, hp).to(device)
        freeze_backbone = False
        if name == "resnet18_finetuned":
            if hp["strategy"] == "feature_extraction":
                for p in model.backbone.parameters():
                    p.requires_grad = False
                freeze_backbone = True
                params = [{"params": model.classifier.parameters(), "lr": hp["lr_head"]}]
            else:  # fine_tuning con learning rate diferencial
                params = [
                    {"params": model.backbone.parameters(), "lr": hp["lr_backbone"]},
                    {"params": model.classifier.parameters(), "lr": hp["lr_head"]},
                ]
        else:
            params = [{"params": model.parameters(), "lr": hp["lr"]}]
        optimizer = torch.optim.AdamW(params, weight_decay=hp["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"])

        history: list[dict[str, float]] = []
        best_key, best_state, best_epoch, bad_epochs = (-1.0, 0.0), None, 0, 0
        t0 = time.time()
        for epoch in range(1, hp["epochs"] + 1):
            tr_loss, tr_acc = self._run_epoch(
                model, train_loader, criterion, device, optimizer, freeze_backbone
            )
            va_loss, va_acc = self._run_epoch(model, valid_loader, valid_criterion, device)
            scheduler.step()
            history.append(
                {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                 "val_loss": va_loss, "val_acc": va_acc}
            )
            logger.info(
                "[%s] epoch %d/%d | train loss %.3f acc %.3f | valid loss %.3f acc %.3f | %.0fs",
                name, epoch, hp["epochs"], tr_loss, tr_acc, va_loss, va_acc, time.time() - t0,
            )
            key = (va_acc, -va_loss)  # mejor checkpoint: mayor accuracy en valid (desempata menor loss)
            if key > best_key:
                best_key, best_epoch, bad_epochs = key, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= hp["patience"]:
                    logger.info("Early stopping en la epoca %d (mejor: %d)", epoch, best_epoch)
                    break

        model.load_state_dict(best_state)
        model.classes = classes
        save_path = Path(hp.get("save_path") or self.active_checkpoint)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.cpu().eval(), save_path)  # modelo completo, no solo state_dict
        self._loaded.pop(name, None)  # fuerza a recargar el checkpoint nuevo

        self.history = history
        self.output_path.mkdir(parents=True, exist_ok=True)
        tag = f"_{hp['tag']}" if hp.get("tag") else ""  # "tag" distingue experimentos del mismo modelo
        (self.output_path / f"{name}{tag}_history.json").write_text(
            json.dumps({"hparams": hp, "best_epoch": best_epoch, "history": history}, indent=2),
            encoding="utf-8",
        )
        logger.info("Checkpoint guardado en %s (mejor epoca %d)", save_path, best_epoch)

    def evaluate_classifier(self, split: str = "test", samples=None) -> dict[str, float]:
        """
        Evalua el modelo activo sobre el conjunto de prueba.

        Debe reportar: accuracy, precision, recall (sensibilidad),
        specificity (especificidad) y F1-Score. La matriz de confusion y las
        curvas de entrenamiento se documentan en 2-train.ipynb.

        Retorna un dict con las metricas, ej:
          {"accuracy": 0.91, "precision": 0.90, "recall": 0.89,
           "specificity": 0.99, "f1": 0.90}

        Por defecto evalua el split `test` de splits.csv. Con `samples` (DataFrame con columnas
        `path` y `breed`) evalua otro conjunto, por ejemplo el independiente. Las metricas son
        macro (promedio entre razas presentes en el conjunto). El detalle (predicciones, matriz de
        confusion) queda en `self.last_eval` para la notebook.
        """
        model = self.load_model()
        if not isinstance(model, torch.nn.Module):
            raise ValueError("evaluate_classifier requiere un checkpoint .pth (PyTorch)")
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        classes = list(getattr(model, "classes", None) or [])
        if not classes:
            raise ValueError("El checkpoint no trae la lista de razas ('classes').")
        class_to_idx = {c: i for i, c in enumerate(classes)}

        if samples is None:
            df, _ = self._load_splits()
            samples = df[df["split"] == split]
        unknown = sorted(set(samples["breed"]) - set(classes))
        if unknown:
            raise ValueError(f"Razas que el modelo no conoce: {unknown}")
        y_true = samples["breed"].map(class_to_idx).to_numpy()

        loader = self._make_loader(
            samples["path"], y_true, eval_transform(self.image_size), 64, shuffle=False
        )
        probs = []
        with torch.no_grad():
            for x, _ in loader:
                probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
        model.cpu()
        probs = np.concatenate(probs)
        y_pred = probs.argmax(1)

        k = len(classes)
        cm = np.zeros((k, k), dtype=np.int64)
        np.add.at(cm, (y_true, y_pred), 1)
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        tn = cm.sum() - tp - fp - fn
        present = cm.sum(1) > 0  # razas que aparecen en este conjunto

        def safe(a, b):
            return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b > 0)

        prec, rec, spec = safe(tp, tp + fp), safe(tp, tp + fn), safe(tn, tn + fp)
        f1 = safe(2 * prec * rec, prec + rec)

        self.last_eval = {
            "classes": classes, "y_true": y_true, "y_pred": y_pred, "probs": probs,
            "paths": list(samples["path"]), "confusion_matrix": cm,
            "per_class": {
                "precision": prec, "recall": rec, "specificity": spec, "f1": f1,
            },
        }
        return {
            "accuracy": float((y_true == y_pred).mean()),
            "precision": float(prec[present].mean()),
            "recall": float(rec[present].mean()),
            "specificity": float(spec[present].mean()),
            "f1": float(f1[present].mean()),
            "n_images": int(len(y_true)),
        }

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando el modelo propio activo
        (penultima capa del ResNet18 fine-tuned o de la CNN custom).

        Se usa cuando EMBEDDING_MODEL != baseline para que la busqueda por
        similitud (Etapa 1) funcione con los modelos entrenados.
        La imagen llega en BGR (OpenCV). Retorna una lista de floats de
        dimension EMBEDDING_DIM.
        """
        model = self.load_model()
        if not isinstance(model, torch.nn.Module) or not hasattr(model, "embed"):
            raise ValueError(
                "extract_custom_embedding requiere un checkpoint .pth con metodo embed() "
                "(ResNet18Classifier o CNNCustom)."
            )
        model.eval()
        tensor = eval_transform(self.image_size)(bgr_to_pil(image)).unsqueeze(0)
        with torch.no_grad():
            emb = model.embed(tensor)[0].numpy().astype(np.float64)
        norm = float(np.linalg.norm(emb))
        return (emb / norm if norm > 0 else emb).tolist()  # norma 1: coseno = producto punto