"""Lecteur PaddleOCR-VL-1.6 : parsing de page, tableau de coordonnées en structure.

PaddleOCR-VL découpe la page en blocs (titres, paragraphes, tableaux, cachets…) et transcrit
chacun ; les tableaux sont rendus en HTML. Il ne sait pas quel texte est le numéro de titre :
ce lecteur en retient surtout le tableau de coordonnées, là où un chiffre faux coûte cher, et
quelques champs à forme fixe (n° de titre, NICAD, superficie) repérés par motif pour
recouper la lecture de Qwen3-VL.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from app.core.config import PaddleSettings
from model.coordinate_table import best_coordinate_table, points_from_text
from app.core.normalize import fix_digits, parse_area
from model.reading import DocumentReading, FieldReading, PointReading
from app.core.schema import BBox, Source

MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"

# Un seul groupe de capture, suffixe optionnel : « 22 665/GR », « 2/C », mais aussi
# « 1835 » que le Kaolack écrit sans suffixe.
_NUM = r"((?:\d[\d .]*\d|\d)(?:\s*/\s*[A-Za-z]{1,4})?)"
_TITLE = re.compile(
    r"(?:T\.?\s?F\.?|titre(?:\s+foncier)?)\s+(?!m[èe]re)"
    r"(?:n\s?[°ºo]\.?|num[ée]ro)?\s*:?\s*" + _NUM,
    re.IGNORECASE,
)
_NICAD = re.compile(r"NICAD\s*(?:n\s?[°ºo]\.?)?\s*:?\s*([\d\s\-]{10,24}\d)", re.IGNORECASE)
_AREA_KEY = re.compile(r"superficie|contenance|surface|contenant", re.IGNORECASE)

# Valeur par défaut : paresseuse, bornée par l’étiquette suivante. Sans cela,
# « Region: Dakar Department: Dakar » donnerait « Dakar Department: Dakar ».
_VALUE = r"[^\n|]{1,60}?(?=\s+[A-ZÉÈÀ][\w’.-]*\s*:|\s*\|\s*|\s*$)"


def _labelled(label: str, value: str = _VALUE) -> re.Pattern:
    """Motif « étiquette : valeur ». L'étiquette tolère les fautes d'OCR courantes
    (accents perdus, lettres confondues) via les classes de caractères."""
    return re.compile(rf"{label}\s*[:\-]?\s*(?:n[°ºo*]\s*)?({value})", re.IGNORECASE)


_MOTHER_TITLE = re.compile(
    r"T\w{1,3}e?\s+M[èeé]re\s*(?:n\s?[°ºo*]\.?)?\s*:?\s*" + _NUM,
    re.IGNORECASE,
)
_LOT = _labelled(r"Lot", _NUM)
_SUBDIVISION = _labelled(r"Lotissement")
_REQUISITION = _labelled(r"R[ée]qu[ei]s[it]{1,2}ion", _NUM)
_LAND_REGISTRY = _labelled(r"LIVRE\s+FONCIER\s+DE")
_REGION = _labelled(r"R[ée]gion")
_DEPARTMENT = _labelled(r"D[ée]part[ea]?ment")
_ARRONDISSEMENT = _labelled(r"Arron?d[is]{0,3}[se]{0,2}[mn]?[ea]?n?t?")
_COMMUNE = _labelled(r"C[ou]mm?une")
_OWNER = _labelled(r"Requ[ée]rant|Propri[ée]taire")
_COORD_SYSTEM = _labelled(r"Syst[èe]me\s+de\s+Coordonn[ée]es")
_SCALE = re.compile(r"ECHELLE\s*[=:]?\s*(1\s*[:/]\s*\d+)", re.IGNORECASE)
_CERTIFIED_ON = re.compile(r"le\s+(\d{1,2}[\s.]+\w{3,10}[\s.]+\d{4})", re.IGNORECASE)
_DISCLAIMER = re.compile(
    r"(ne\s+constitue\s+pas\s+un\s+titre\s+de\s+propri[ée]t[ée])", re.IGNORECASE
)

# Un champ = un motif. L'ordre n'a pas d'importance, chaque champ est cherché une fois.
_FIELD_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("title_number", _TITLE),
    ("mother_title", _MOTHER_TITLE),
    ("lot", _LOT),
    ("subdivision", _SUBDIVISION),
    ("requisition", _REQUISITION),
    ("land_registry", _LAND_REGISTRY),
    ("nicad", _NICAD),
    ("region", _REGION),
    ("department", _DEPARTMENT),
    ("arrondissement", _ARRONDISSEMENT),
    ("commune", _COMMUNE),
    ("owner", _OWNER),
    ("coordinate_system", _COORD_SYSTEM),
    ("scale", _SCALE),
    ("certified_on", _CERTIFIED_ON),
    ("disclaimer", _DISCLAIMER),
)


def _bbox(b: Any) -> BBox | None:
    try:
        x0, y0, x1, y1 = (float(v) for v in b)
        return BBox(x0=x0, y0=y0, x1=x1, y1=y1)
    except (TypeError, ValueError):
        return None


def _blocks(data: dict) -> list[dict]:
    blocks = data.get("parsing_res_list") or []
    # Ordre de lecture quand il est fourni, sinon ordre de la liste.
    return sorted(blocks, key=lambda b: (b.get("block_order") is None, b.get("block_order") or 0))


def page_text(data: dict) -> str:
    return "\n".join(
        str(b.get("block_content") or "") for b in _blocks(data) if b.get("block_label") != "table"
    )

_TAGS = re.compile(r"<[^>]+>")

def _plain_text(content: str) -> str:
    """Contenu d'un bloc en texte lisible : les balises deviennent des espaces, afin que
    « <td>Superficie</td><td>02a 55ca</td> » se lise « Superficie 02a 55ca »."""
    import html

    text = _TAGS.sub(" ", content)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()

def _fields_from_text(data: dict) -> dict[str, FieldReading]:
    fields: dict[str, FieldReading] = {}
    blocks = _blocks(data)
    # Le bloc identifié comme titre par la mise en page passe en premier : le dessin du
    # plan porte « Surplus du TF 976/GR », qui ne doit pas devenir le numéro du titre.
    ordered = [b for b in blocks if b.get("block_label") == "doc_title"]
    ordered += [b for b in blocks if b.get("block_label") != "doc_title"]
    for block in ordered:
        # Un champ peut se trouver dans n'importe quelle région, y compris dans un
        # encadré que la mise en page classe « table » (cas du cadre du plan).
        text = fix_digits(_plain_text(str(block.get("block_content") or "")))
        bbox = _bbox(block.get("block_bbox"))
        for name, pattern in _FIELD_PATTERNS:
            if name in fields:
                continue
            if m := pattern.search(text):
                value = (m.group(1) if m.lastindex else m.group(0)).strip(" .:-")
                if value:                      # une case vide ne produit pas de champ
                    fields[name] = FieldReading(raw=value, bbox=bbox)
        if "area" not in fields and (k := _AREA_KEY.search(text)):
            tail = text[k.end() : k.end() + 60]
            if parse_area(tail) is not None:
                fields["area"] = FieldReading(raw=tail.strip(), bbox=bbox)
    return fields


def reading_from_result(data: dict) -> DocumentReading:
    """Sortie JSON de PaddleOCR-VL (`res.json["res"]`) → lecture brute. Fonction pure."""
    tables = [b for b in _blocks(data) if b.get("block_label") == "table"]
    table = best_coordinate_table([str(b.get("block_content") or "") for b in tables])
    points: list[PointReading]
    sides = []
    if table is not None:
        points, sides = table.points, table.sides
        # La cellule exacte n'est pas localisée par le pipeline : on rattache le bloc tableau.
        tbbox = next((_bbox(b.get("block_bbox")) for b in tables), None)
        for p in points:
            p.bbox = tbbox
    else:
        points = points_from_text(page_text(data))
    pre = data.get("doc_preprocessor_res") or {}
    angle = pre.get("angle") if isinstance(pre, dict) else None
    return DocumentReading(
        source=Source.PADDLE_VL,
        model=MODEL_ID,
        fields=_fields_from_text(data),
        points=points,
        sides=sides,
        raw_output={
            k: v
            for k, v in data.items()
            if k
            in ("parsing_res_list", "doc_preprocessor_res", "width", "height", "model_settings")
        },
        orientation_deg=angle if isinstance(angle, int) and angle >= 0 else None,
    )


class PaddleVLReader:
    source = Source.PADDLE_VL
    model_id = MODEL_ID

    def __init__(self, settings: PaddleSettings | None = None) -> None:
        self.settings = settings or PaddleSettings()
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            try:
                from paddleocr import PaddleOCRVL
            except ImportError as exc:  # pragma: no cover - dépend de l'environnement GPU
                raise RuntimeError("PaddleOCR-VL non installé : voir requirements-gpu.txt") from exc
            s = self.settings
            kwargs: dict[str, Any] = {
                "pipeline_version": s.pipeline_version,
                "use_doc_orientation_classify": s.use_doc_orientation_classify,
                "use_doc_unwarping": s.use_doc_unwarping,
            }
            kwargs.update(
                vl_rec_backend=s.server_kind,
                vl_rec_server_url=s.server_url,
                layout_merge_bboxes_mode=s.layout_merge_bboxes_mode,
                merge_layout_blocks=s.merge_layout_blocks,
            )
            if s.device:
                kwargs["device"] = s.device
            self._pipeline = PaddleOCRVL(**kwargs)
        return self._pipeline

    def read(self, image: np.ndarray) -> DocumentReading:
        (res,) = list(self.pipeline.predict(image))[:1]
        data = res.json.get("res", res.json)
        reading = reading_from_result(data)
        # Image réorientée par PaddleOCR : c'est elle que doit lire Qwen3-VL.
        try:
            out = res["doc_preprocessor_res"]["output_img"]
            reading.image = np.ascontiguousarray(out) if out is not None else None
        except (KeyError, TypeError):
            reading.image = None
        return reading
