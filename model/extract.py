"""Point d'entrée du modèle : image -> Extraction."""
from __future__ import annotations

from pathlib import Path

from model.ocr import load_image, read_boxes
from model.parse import parse
from model.schema import Extraction


def extract(image_path: Path, document_id: str) -> Extraction:
    return parse(read_boxes(load_image(image_path)), document_id)
