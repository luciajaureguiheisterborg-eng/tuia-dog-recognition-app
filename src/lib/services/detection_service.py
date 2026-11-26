from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch

from lib.preprocessing import bgr_to_pil, eval_transform
from lib.schemas import ClassifyResult, DetectResult, DogDetection
from lib.services.classifier_service import ClassifierService

logger = logging.getLogger(__name__)


class DetectionService:
    """Etapa 3: pipeline de deteccion y clasificacion.

    Funciones implementadas:
      - detect_dogs(image)
      - classify_detected_dog(crop)

    La orquestacion (predict: deteccion -> recorte -> clasificacion -> JSON)
    ya esta provista.
    """

    def __init__(
        self,
        classifier: ClassifierService,
        yolo_model: str,
        conf_threshold: float,
        dog_class_id: int,
    ) -> None:
        self.classifier = classifier
        self.yolo_model_name = yolo_model
        self.conf_threshold = conf_threshold
        self.dog_class_id = dog_class_id

    @staticmethod
    def _clip_xyxy(
        x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)
        return x1, y1, x2, y2

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV / ultralytics)
        return image

    # ------------------------------------------------------------------
    # Etapa 3: funciones a implementar
    # ------------------------------------------------------------------

    def _get_detector(self):
        """Carga YOLO una sola vez (lazy) y lo reutiliza en las siguientes llamadas."""
        detector = getattr(self, "_detector", None)
        if detector is None:
            from ultralytics import YOLO  # import tardio: solo hace falta en la Etapa 3

            logger.info("Loading YOLO detector: %s", self.yolo_model_name)
            detector = YOLO(self.yolo_model_name)
            self._detector = detector
        return detector

    def detect_dogs(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        """
        Detecta todos los perros presentes en la imagen usando un modelo YOLO
        pre-entrenado (ej: YOLOv8n via ultralytics). No es necesario entrenar
        el detector.

        Sugerencias:
          - self.yolo_model_name, self.conf_threshold y self.dog_class_id
            (clase 'dog' = 16 en COCO) vienen de la configuracion (.env).
          - Debe funcionar con un perro, multiples perros y escenas complejas.

        Retorna una lista de ((x1, y1, x2, y2), confidence) en pixeles.
        """
        if image is None or image.size == 0:
            return []
        # ultralytics interpreta un np.ndarray como BGR (igual que cv2.imread).
        results = self._get_detector().predict(
            image,
            conf=self.conf_threshold,
            classes=[self.dog_class_id],
            verbose=False,
        )
        detections: list[tuple[tuple[int, int, int, int], float]] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if int(box.cls[0]) != self.dog_class_id:
                    continue
                x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
                detections.append(((x1, y1, x2, y2), float(box.conf[0])))
        detections.sort(key=lambda d: d[1], reverse=True)
        return detections

    def classify_detected_dog(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Clasifica la raza del recorte de un perro detectado usando el modelo
        entrenado en la Etapa 2 (self.classifier.load_model()).

        El recorte llega en BGR (OpenCV). Retorna (raza, score).
        """
        if crop is None or crop.size == 0:
            return "unknown", 0.0

        model = self.classifier.load_model()
        # Mismo pipeline que valid/test: (1, 3, H, W)
        tensor = eval_transform(self.classifier.image_size)(bgr_to_pil(crop)).unsqueeze(0)

        if isinstance(model, torch.nn.Module):
            model.eval()
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs))
            score = float(probs[idx])
            classes = getattr(model, "classes", None)
        else:  # onnxruntime.InferenceSession
            batch = tensor.numpy().astype(np.float32)
            input_name = model.get_inputs()[0].name
            logits = model.run(None, {input_name: batch})[0][0]
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
            idx = int(np.argmax(probs))
            score = float(probs[idx])
            meta = model.get_modelmeta().custom_metadata_map
            classes = json.loads(meta["classes"]) if "classes" in meta else None

        if not classes:
            raise ValueError(
                "El checkpoint no trae la lista de razas ('classes'). "
                "Guardala junto con el modelo al entrenar."
            )
        return str(classes[idx]), score

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def classify_image(
        self, source_path: str, output_path: Path, model_name: str | None = None
    ) -> str:
        """Clasifica la imagen completa con el modelo entrenado (pestaña Etapa 2).

        Reutiliza classify_detected_dog tratando la imagen entera como recorte,
        por lo que requiere la Etapa 2 (modelo entrenado) y classify_detected_dog.
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        # El modelo elegido vale solo para esta clasificacion: se restaura el activo para
        # que /detect siga usando el modelo configurado.
        previous = self.classifier.active_model_name
        if model_name:
            self.classifier.set_active_model(model_name)
        try:
            breed, score = self.classify_detected_dog(image)
        finally:
            self.classifier.set_active_model(previous)
        payload = ClassifyResult(
            source_path=source_path,
            model=model_name or self.classifier.active_model_name,
            breed=breed,
            score=round(float(score), 4),
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)

    def predict(self, source_path: str, output_path: Path) -> str:
        """Flujo completo: deteccion -> bounding boxes -> recortes -> clasificacion.

        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        height, width = image.shape[:2]

        detections: list[DogDetection] = []
        for (box, det_score) in self.detect_dogs(image):
            x1, y1, x2, y2 = self._clip_xyxy(*[int(v) for v in box], height, width)
            crop = image[y1:y2, x1:x2]
            breed, breed_score = self.classify_detected_dog(crop)
            detections.append(
                DogDetection(
                    bbox=[x1, y1, x2, y2],
                    det_score=round(float(det_score), 4),
                    breed=breed,
                    breed_score=round(float(breed_score), 4),
                )
            )

        detected_breeds = sorted({item.breed for item in detections if item.breed != "unknown"})
        payload = DetectResult(
            source_path=source_path,
            detections=detections,
            detected_breeds=detected_breeds,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)