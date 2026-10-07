"""Mesure : compare une extraction à son annotation, champ par champ.

Quatre issues possibles pour chaque valeur attendue :
  EXACT   identique à la vérité terrain
  NEAR    à un seul caractère près (erreur typique d'OCR, mais fausse quand même)
  WRONG   lue, mais différente
  MISSING non lue
Une valeur NEAR est comptée comme une erreur dans la calibration : un chiffre faux dans une
coordonnée déplace le sommet.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from model.annotation import Annotation
from model.schema import Extracted, Extraction


class Outcome(str, Enum):
    EXACT = "exact"
    NEAR = "near"
    WRONG = "wrong"
    MISSING = "missing"


@dataclass(frozen=True)
class FieldResult:
    document_id: str
    field: str                 # « title_number », « points[2].x », …
    expected: str
    got: str | None
    confidence: float | None
    outcome: Outcome

    @property
    def correct(self) -> bool:
        return self.outcome is Outcome.EXACT


@dataclass(frozen=True)
class DocReport:
    document_id: str
    results: list[FieldResult]
    spurious_points: int       # points extraits en plus de ceux de la vérité terrain


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().lower()


def _judge(expected: str, got: str | None, *, exact_only: bool = False) -> Outcome:
    if got is None:
        return Outcome.MISSING
    if got == expected:
        return Outcome.EXACT
    if not exact_only and levenshtein(expected, got) == 1:
        return Outcome.NEAR
    return Outcome.WRONG


def _fmt_coord(value: float | None, decimals: int) -> str | None:
    return None if value is None else f"{value:.{decimals}f}"


def _result(doc: str, field: str, expected: str, ex: Extracted, got: str | None, **kw) -> FieldResult:
    return FieldResult(doc, field, expected, got, ex.confidence if ex.present else None,
                       _judge(expected, got, **kw))


def compare(extraction: Extraction, truth: Annotation) -> DocReport:
    """Compare une extraction à sa vérité terrain. Les points sont alignés par rang dans le tableau."""
    doc = truth.document_id
    out: list[FieldResult] = []

    for name in ("title_number", "nicad", "locality"):
        expected = getattr(truth, name)
        if expected is None:
            continue
        ex: Extracted = getattr(extraction, name)
        got = _norm_text(ex.value) if ex.present else None
        out.append(_result(doc, name, _norm_text(expected), ex, got, exact_only=True))

    if truth.printed_area is not None:
        ex = extraction.printed_area
        got = str(int(ex.value.m2)) if ex.present else None
        out.append(_result(doc, "printed_area_m2", str(int(truth.printed_area.m2)), ex, got,
                           exact_only=True))

    by_row = {p.row_index: p for p in extraction.points}
    for i, tp in enumerate(truth.points):
        got_pt = by_row.get(i)
        for axis in ("x", "y"):
            expected = getattr(tp, axis)
            decimals = truth.decimals if truth.decimals is not None else len(expected.split(".")[1])
            if got_pt is None:
                ex = Extracted[float]()
                got = None
            else:
                ex = getattr(got_pt, axis)
                got = _fmt_coord(ex.value, decimals)
            out.append(_result(doc, f"points[{i}].{axis}", expected, ex, got))

    spurious = sum(1 for r in by_row if r >= len(truth.points))
    return DocReport(doc, out, spurious)


# --------------------------------------------------------------------------- agrégation
def summarize(reports: list[DocReport]) -> dict[str, object]:
    results = [r for rep in reports for r in rep.results]
    n = len(results)
    c = Counter(r.outcome for r in results)
    return {
        "documents": len(reports),
        "values": n,
        **{o.value: c.get(o, 0) for o in Outcome},
        "exact_rate": c.get(Outcome.EXACT, 0) / n if n else 0.0,
        "spurious_points": sum(rep.spurious_points for rep in reports),
    }


# --------------------------------------------------------------------------- calibration
def calibration_table(results: list[FieldResult], edges: tuple[float, ...] = (0.0, 0.5, 0.7, 0.85, 0.95, 1.0001)):
    """Taux d'erreur par tranche de confiance. Ne compte que les valeurs lues (confiance connue)."""
    rows = []
    scored = [r for r in results if r.confidence is not None]
    for lo, hi in zip(edges, edges[1:], strict=False):
        sel = [r for r in scored if lo <= r.confidence < hi]
        errors = sum(1 for r in sel if not r.correct)
        rows.append({"from": lo, "to": min(hi, 1.0), "n": len(sel), "errors": errors,
                     "error_rate": errors / len(sel) if sel else None})
    return rows


def threshold_for(results: list[FieldResult], max_error_rate: float = 0.01) -> dict[str, float | None]:
    """Plus petit seuil dont les valeurs acceptées (confiance >= seuil) respectent le taux d'erreur visé.

    Retourne aussi la part de valeurs envoyées en relecture (sous le seuil ou non lues).
    Résultat peu fiable sur un petit échantillon : à relancer quand le jeu annoté grandit.
    """
    total = len(results)
    scored = sorted((r for r in results if r.confidence is not None), key=lambda r: r.confidence)
    for cand in sorted({r.confidence for r in scored}):
        accepted = [r for r in scored if r.confidence >= cand]
        errors = sum(1 for r in accepted if not r.correct)
        if accepted and errors / len(accepted) <= max_error_rate:
            return {"threshold": cand, "accepted": len(accepted),
                    "review_share": 1 - len(accepted) / total if total else None}
    return {"threshold": None, "accepted": 0, "review_share": 1.0 if total else None}