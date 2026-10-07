"""Lecture OCR brute : image -> boîtes de texte. Aucune interprétation ici (voir parse.py)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OcrBox:
    text: str
    confidence: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def load_image(path: Path) -> np.ndarray:
    """Charge une image en BGR. OpenCV d'abord, Pillow en secours (webp, avif, jfif selon la build)."""
    import cv2

    img = cv2.imread(str(path))
    if img is not None:
        return img
    from PIL import Image

    with Image.open(path) as im:
        rgb = np.array(im.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


_ENGINE = None


def read_boxes(image: np.ndarray) -> list[OcrBox]:
    """RapidOCR sur l'image telle quelle.

    L'essai de base a montré que l'image brute donne de meilleurs résultats que les variantes
    agrandies ou binarisées : on ne prétraite donc pas à ce stade.
    """
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        _ENGINE = RapidOCR()
    result, _ = _ENGINE(image)
    boxes: list[OcrBox] = []
    for poly, text, conf in result or []:
        pts = np.asarray(poly, dtype=float)
        boxes.append(OcrBox(str(text), float(conf), *pts.min(axis=0), *pts.max(axis=0)))
    return boxes
