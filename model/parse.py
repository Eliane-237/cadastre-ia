# texte → points, nom, superficie, identifiants

"""Interprétation des boîtes OCR : reconstruit le tableau de coordonnées et la superficie.

Fonction pure (aucun OCR, aucune image) donc testable sans moteur. Hypothèses de cette version :
un seul tableau de coordonnées, lu en lignes ; deux colonnes de nombres (Est puis Nord).
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from model.ocr import OcrBox
from model.schema import (
    Area, BBox, BoundaryPoint, Extracted, Extraction, Source,
)

_COORD = re.compile(r"\d{5,8}\.\d{1,3}")
_LABEL = re.compile(r"^[A-Za-z]{0,3}[\s.\-]?\d{1,3}$")
# Les largeurs de l'image varient trop pour fixer des seuils en pixels : tout est relatif à la
# hauteur médiane des boîtes de texte.
ROW_TOLERANCE = 0.6


def normalize(text: str) -> str:
    """Virgule décimale -> point, espaces autour du point supprimés."""
    text = text.replace(",", ".")
    return re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", text)


@dataclass
class _Token:
    value: str
    confidence: float
    x0: float
    y0: float
    x1: float
    y1: float
    column: str = ""        # « x » ou « y », fixé plus tard

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def _tokens(boxes: list[OcrBox]) -> list[_Token]:
    out: list[_Token] = []
    for b in boxes:
        found = _COORD.findall(normalize(b.text))
        k = len(found)
        w = b.x1 - b.x0
        for i, v in enumerate(found):   # plusieurs nombres dans une même boîte : on la découpe
            out.append(_Token(v, b.confidence, b.x0 + w * i / k, b.y0, b.x0 + w * (i + 1) / k, b.y1))
    return out


def _assign_columns(tokens: list[_Token]) -> None:
    """Répartit les jetons en colonnes Est/Nord d'après leur position horizontale."""
    # Amorce : 6 chiffres avant le point = Est, 7 = Nord (Sénégal, UTM 28N). Les jetons mal lus
    # (8 chiffres, 5 chiffres) sont rattachés ensuite à la colonne la plus proche.
    seeds_x = [t.cx for t in tokens if len(t.value.split(".")[0]) == 6]
    seeds_y = [t.cx for t in tokens if len(t.value.split(".")[0]) == 7]
    if seeds_x and seeds_y:
        cx_x, cx_y = statistics.median(seeds_x), statistics.median(seeds_y)
    else:
        xs = sorted(t.cx for t in tokens)
        mid = statistics.median(xs)
        left = [v for v in xs if v <= mid] or xs
        right = [v for v in xs if v > mid] or xs
        cx_x, cx_y = statistics.median(left), statistics.median(right)
    for t in tokens:
        t.column = "x" if abs(t.cx - cx_x) <= abs(t.cx - cx_y) else "y"


def _group_rows(tokens: list[_Token], tol: float) -> list[list[_Token]]:
    rows: list[list[_Token]] = []
    for t in sorted(tokens, key=lambda t: t.cy):
        if rows and abs(t.cy - statistics.mean(r.cy for r in rows[-1])) <= tol:
            rows[-1].append(t)
        else:
            rows.append([t])
    return rows


def _extracted(t: _Token | None) -> Extracted[float]:
    if t is None:
        return Extracted[float]()
    return Extracted[float](value=float(t.value), confidence=t.confidence, source=Source.OCR,
                            raw_text=t.value, bbox=BBox(x0=t.x0, y0=t.y0, x1=t.x1, y1=t.y1))


def _best(row: list[_Token], column: str) -> _Token | None:
    cands = [t for t in row if t.column == column]
    return max(cands, key=lambda t: t.confidence) if cands else None


def _label(row: list[_Token], boxes: list[OcrBox], tol: float, default: str) -> str:
    row_cy = statistics.mean(t.cy for t in row)
    left = min(t.x0 for t in row)
    cands = [b for b in boxes if b.x1 <= left and abs(b.cy - row_cy) <= tol
             and _LABEL.match(b.text.strip()) and not _COORD.search(normalize(b.text))]
    return max(cands, key=lambda b: b.x1).text.strip() if cands else default


def parse_points(boxes: list[OcrBox]) -> list[BoundaryPoint]:
    toks = _tokens(boxes)
    if not toks or not boxes:
        return []
    tol = ROW_TOLERANCE * statistics.median(b.height for b in boxes)
    _assign_columns(toks)
    points: list[BoundaryPoint] = []
    for row in _group_rows(toks, tol):
        i = len(points)
        points.append(BoundaryPoint(label=_label(row, boxes, tol, f"P{i + 1}"), row_index=i,
                                    x=_extracted(_best(row, "x")), y=_extracted(_best(row, "y"))))
    return points


_AREA_FULL = re.compile(r"(\d{1,3})\s*ha\s*(\d{1,2})\s*a\s*(\d{1,2})\s*ca", re.I)
_AREA_SHORT = re.compile(r"(\d{1,2})\s*a\s*(\d{1,2})\s*ca", re.I)


def parse_area(boxes: list[OcrBox]) -> Extracted[Area]:
    """Superficie imprimée « 00 ha 06 a 49 ca ». Confiance = la plus basse des boîtes concernées."""
    ordered = sorted(boxes, key=lambda b: (round(b.cy / max(b.height, 1)), b.x0))
    spans, text = [], ""
    for b in ordered:
        spans.append((len(text), len(text) + len(b.text), b))
        text += b.text + " "
    for rx, with_ha in ((_AREA_FULL, True), (_AREA_SHORT, False)):
        m = rx.search(text)
        if m:
            ha, a, ca = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if with_ha \
                else (0, int(m.group(1)), int(m.group(2)))
            hit = [b for s, e, b in spans if s < m.end() and e > m.start()]
            return Extracted[Area](value=Area(ha=ha, a=a, ca=ca),
                                   confidence=min(b.confidence for b in hit), source=Source.PARSER,
                                   raw_text=m.group(0))
    return Extracted[Area]()


def parse(boxes: list[OcrBox], document_id: str) -> Extraction:
    """Titre, NICAD, propriétaire et localité ne sont pas encore extraits (champs laissés vides)."""
    return Extraction(document_id=document_id, points=parse_points(boxes), printed_area=parse_area(boxes))
