"""Module 1 — acquisition : amener un document en mémoire, et rien d'autre.

Photo, scan ou PDF : la sortie est toujours une liste de pages en BGR. Aucune correction
n'est faite ici — c'est l'affaire de `preprocess`. Séparer les deux permet de rejouer un
prétraitement sans relire le fichier, et de savoir lequel des deux a abîmé une image.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

PDF_DPI = 300
# Écart-type relatif du fond (feuille) au-delà duquel on corrige l'éclairage (à calibrer).


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image(path: Path) -> np.ndarray:
    """Charge une image en BGR. OpenCV d'abord (chemins non ASCII compris), Pillow en secours
    (webp, avif, jfif selon la build)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is not None:
        return img
    from PIL import Image

    with Image.open(path) as im:
        rgb = np.array(im.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_pages(path: Path) -> list[np.ndarray]:
    """Toutes les pages d'un PDF (rendues à 300 dpi), ou l'image seule."""
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        return [load_image(path)]
    import pymupdf

    pages = []
    with pymupdf.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=PDF_DPI, colorspace=pymupdf.csRGB)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            pages.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return pages
