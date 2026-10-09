"""Chargement des documents et prétraitement d'image, avant les deux lecteurs.

PaddleOCR-VL corrige l'orientation (90/180/270°) mais pas la perspective d'une photo ni un
éclairage inégal. On traite ici ces deux défauts, et seulement eux : l'essai de base a montré
que l'image brute lit mieux que les variantes agrandies ou binarisées. Chaque étape n'est
appliquée que si elle est détectée comme nécessaire, et elle est consignée pour l'audit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PDF_DPI = 300
# Écart-type relatif du fond (feuille) au-delà duquel on corrige l'éclairage (à calibrer).
ILLUMINATION_THRESHOLD = 0.08
MAX_SKEW_DEG = 15.0
MIN_SKEW_DEG = 0.3


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


# --------------------------------------------------------------------------- perspective


def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def find_document_quad(image: np.ndarray) -> np.ndarray | None:
    """Repère la feuille dans une photo (quadrilatère dominant). Un scan qui occupe déjà toute
    l'image ne renvoie rien : on ne recadre pas ce qui n'en a pas besoin."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 1000 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(cv2.GaussianBlur(small, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = small.shape[0] * small.shape[1]
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if not 0.3 * area_img < cv2.contourArea(cnt) < 0.95 * area_img:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return _order_quad(approx / scale)
    return None


def warp_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------- éclairage


def _background(gray: np.ndarray) -> np.ndarray:
    """Fond estimé : le texte (sombre, fin) disparaît sous une fermeture puis un flou larges."""
    k = max(gray.shape) // 40 | 1
    closed = cv2.morphologyEx(
        gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    )
    return cv2.GaussianBlur(closed, (0, 0), k / 2).astype(np.float32)


def illumination_unevenness(image: np.ndarray) -> float:
    """Écart-type relatif du fond : ~0 pour un scan, élevé pour une photo avec ombre."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = _background(small)
    return float(bg.std() / max(bg.mean(), 1.0))


def flatten_illumination(image: np.ndarray) -> np.ndarray:
    """Divise chaque canal par le fond estimé : ombres et dégradés disparaissent, les couleurs
    (tampons, encre) sont conservées, sans binarisation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg = np.maximum(_background(gray), 1.0)
    target = np.percentile(bg, 95)
    out = image.astype(np.float32) * (target / bg)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- inclinaison


def estimate_skew(gray: np.ndarray) -> float:
    """Inclinaison du texte en degrés (positif = sens anti-horaire), par Hough sur les lignes
    de texte fusionnées ; médiane pondérée par la longueur."""
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(gray.shape[1] // 60, 5), 1))
    merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    lines = cv2.HoughLinesP(
        merged, 1, np.pi / 1800, threshold=100, minLineLength=gray.shape[1] // 8, maxLineGap=10
    )
    if lines is None:
        return 0.0
    angles, weights = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = np.degrees(np.arctan2(y1 - y2, x2 - x1))
        if abs(angle) <= MAX_SKEW_DEG:
            angles.append(angle)
            weights.append(np.hypot(x2 - x1, y2 - y1))
    if not angles:
        return 0.0
    order = np.argsort(angles)
    cum = np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(angles)[order][np.searchsorted(cum, cum[-1] / 2)])


def rotate(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -angle_deg, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2 - w / 2
    matrix[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(
        image, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
    )


# --------------------------------------------------------------------------- chaîne


def preprocess(
    image: np.ndarray,
    *,
    perspective: bool = True,
    illumination: bool = True,
    deskew: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Applique uniquement les corrections nécessaires ; renvoie l'image et le journal."""
    steps: dict[str, Any] = {"input_shape": list(image.shape[:2])}
    if perspective:
        quad = find_document_quad(image)
        steps["perspective"] = quad.round(1).tolist() if quad is not None else None
        if quad is not None:
            image = warp_quad(image, quad)
    if illumination:
        unevenness = illumination_unevenness(image)
        steps["illumination_unevenness"] = round(unevenness, 4)
        steps["illumination_corrected"] = unevenness > ILLUMINATION_THRESHOLD
        if unevenness > ILLUMINATION_THRESHOLD:
            image = flatten_illumination(image)
    if deskew:
        angle = estimate_skew(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        steps["skew_deg"] = round(angle, 3)
        if abs(angle) >= MIN_SKEW_DEG:
            image = rotate(image, angle)
    steps["output_shape"] = list(image.shape[:2])
    return image, steps


def encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("encodage PNG impossible")
    return buf.tobytes()


def limit_size(image: np.ndarray, max_side: int) -> np.ndarray:
    side = max(image.shape[:2])
    if side <= max_side:
        return image
    f = max_side / side
    return cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
