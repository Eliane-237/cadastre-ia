"""Fusion des deux lectures : consensus, arbitrage géométrique, sinon revue humaine.

Règles, valeur par valeur :
- les deux lecteurs donnent la même chaîne (après normalisation) → consensus, confiance haute ;
- un seul lecteur a lu la valeur → elle est reprise, confiance moyenne ;
- les lecteurs divergent sur des coordonnées → on teste les combinaisons possibles et on garde
  la seule qui rende le document cohérent (surface de Gauss, longueurs des côtés imprimées) ;
  s'il n'y en a aucune, ou plusieurs, la géométrie ne tranche pas : la valeur de
  PaddleOCR-VL (lecteur spécialisé des tableaux) est gardée avec une confiance basse, et le
  document part en revue humaine.

Rien n'est jamais « corrigé » en silence : toutes les lectures restent dans `readings`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from shapely.geometry import Polygon

from model.config import SETTINGS, Settings
from model.numbers import (
    canonical_coordinate,
    canonical_label,
    canonical_length,
    canonical_nicad,
    canonical_text,
    canonical_title,
    parse_area,
)
from model.readers.base import DocumentReading, FieldReading, PointReading, SideReading
from model.schema import (
    Area,
    BoundaryPoint,
    Extracted,
    Extraction,
    Issue,
    Reading,
    Severity,
    Source,
)
from model.validate import shoelace_area, side_length

# Lecteur dont la valeur est gardée par défaut en cas de désaccord non tranché.
TABLE_READER = Source.PADDLE_VL
FIELD_READER = Source.QWEN_VL


@dataclass
class _Candidate:
    source: Source
    raw: str | None
    normalized: str | None
    confidence: float | None = None
    bbox: object = None

    def reading(self) -> Reading:
        return Reading(
            source=self.source,
            raw_text=self.raw,
            normalized=self.normalized,
            confidence=self.confidence,
            bbox=self.bbox,
        )  # type: ignore[arg-type]


@dataclass
class _Conflict:
    point: int
    axis: str
    options: list[_Candidate]


@dataclass
class FusionReport:
    issues: list[Issue] = field(default_factory=list)
    conflicts_total: int = 0
    conflicts_resolved: int = 0


def _single_conf(c: _Candidate, cap: float, cfg: Settings) -> float:
    """Une seule lecture : sa propre confiance si le lecteur en donne une, plafonnée."""
    own = c.confidence if c.confidence is not None else cfg.conf_single
    return min(own, cap)


def _decide(
    cands: list[_Candidate], cfg: Settings, prefer: Source, single_cap: float | None = None
) -> tuple[_Candidate | None, Source, float, bool]:
    """(retenue, source, confiance, désaccord) pour une valeur lue par 0, 1 ou 2 lecteurs.

    `single_cap` plafonne la confiance d'une lecture unique : pour une coordonnée, une seule
    lecture n'est jamais suffisante, même si le modèle se dit sûr de lui."""
    cap = cfg.conf_single_cap if single_cap is None else single_cap
    usable = [c for c in cands if c.normalized is not None]
    if not usable:
        return None, prefer, 0.0, False
    if len(usable) == 1:
        return usable[0], usable[0].source, _single_conf(usable[0], cap, cfg), False
    if len({c.normalized for c in usable}) == 1:
        return usable[0], Source.CONSENSUS, cfg.conf_agree, False
    chosen = next((c for c in usable if c.source is prefer), usable[0])
    return chosen, chosen.source, cfg.conf_conflict, True


# --------------------------------------------------------------------------- points


def _align(
    a: list[PointReading], b: list[PointReading]
) -> list[tuple[PointReading | None, PointReading | None]]:
    """Aligne par libellé quand les libellés sont lisibles et uniques, sinon par rang."""
    la = [canonical_label(p.label) for p in a]
    lb = [canonical_label(p.label) for p in b]
    usable = (
        all(la)
        and all(lb)
        and len(set(la)) == len(la)
        and len(set(lb)) == len(lb)
        and len(set(la) & set(lb)) >= max(1, min(len(la), len(lb)) // 2)
    )
    if not usable:
        return list(itertools.zip_longest(a, b))
    by_b = dict(zip(lb, b, strict=True))
    pairs: list[tuple[PointReading | None, PointReading | None]] = [
        (p, by_b.pop(lab, None)) for p, lab in zip(a, la, strict=True)
    ]
    pairs.extend((None, p) for p in by_b.values())  # points lus seulement par b, en fin
    return pairs


def _coord_cands(pair, axis: str, sources: tuple[Source, Source]) -> list[_Candidate]:
    out = []
    for p, src in zip(pair, sources, strict=True):
        if p is None:
            continue
        raw = getattr(p, axis)
        out.append(
            _Candidate(
                src, raw, canonical_coordinate(raw), getattr(p, f"{axis}_confidence"), p.bbox
            )
        )
    return out


def _extracted_coord(
    chosen: _Candidate | None, source: Source, conf: float, cands: list[_Candidate]
) -> Extracted[float]:
    return Extracted[float](
        value=float(chosen.normalized) if chosen and chosen.normalized else None,
        confidence=conf if chosen else 0.0,
        source=source,
        raw_text=chosen.raw if chosen else None,
        bbox=chosen.bbox if chosen else None,  # type: ignore[arg-type]
        readings=[c.reading() for c in cands],
    )


# --------------------------------------------------------------------------- côtés


def _side_index(s: SideReading, labels: list[str | None], n: int) -> int | None:
    if s.from_label and s.to_label:
        f, t = canonical_label(s.from_label), canonical_label(s.to_label)
        for i in range(n):
            a, b = labels[i], labels[(i + 1) % n]
            if {a, b} == {f, t}:
                return i
        return None
    if s.row_index is not None and 0 <= s.row_index < n:
        return s.row_index
    return None


def _fuse_sides(
    readings: list[DocumentReading], labels: list[str | None], n: int, cfg: Settings
) -> list[Extracted[float]]:
    per_side: dict[int, list[_Candidate]] = {}
    for r in readings:
        for s in r.sides:
            i = _side_index(s, labels, n)
            if i is not None:
                per_side.setdefault(i, []).append(
                    _Candidate(r.source, s.length, canonical_length(s.length), s.confidence)
                )
    sides = []
    for i in range(n):
        cands = per_side.get(i, [])
        chosen, source, conf, _ = _decide(cands, cfg, TABLE_READER,
                                          cfg.conf_single_coordinate_cap)
        sides.append(
            Extracted[float](
                value=float(chosen.normalized) if chosen and chosen.normalized else None,
                confidence=conf,
                source=source,
                raw_text=chosen.raw if chosen else None,
                readings=[c.reading() for c in cands],
            )
        )
    return sides


# --------------------------------------------------------------------------- champs


_FIELD_NORMALIZERS = {
    "title_number": canonical_title,
    "nicad": canonical_nicad,
    "owner": canonical_text,
    "locality": canonical_text,
}


def _field_cands(readings: list[DocumentReading], name: str) -> list[_Candidate]:
    out = []
    for r in readings:
        f: FieldReading = r.field(name)
        if f.raw is None:
            continue
        if name == "area":
            area = parse_area(f.raw)
            norm = str(int(area.m2)) if area else None
        else:
            norm = _FIELD_NORMALIZERS[name](f.raw)
        out.append(_Candidate(r.source, f.raw, norm, f.confidence, f.bbox))
    return out


def _fuse_field(
    readings: list[DocumentReading], name: str, cfg: Settings, report: FusionReport
) -> Extracted:
    cands = _field_cands(readings, name)
    chosen, source, conf, conflict = _decide(cands, cfg, FIELD_READER)
    if conflict:
        report.issues.append(
            Issue(
                code="FIELD_CONFLICT",
                severity=Severity.WARNING,
                message=f"{name} : lectures divergentes "
                f"({' / '.join(repr(c.raw) for c in cands)}).",
            )
        )
    if name == "area":
        value = parse_area(chosen.raw) if chosen else None
        return Extracted[Area](
            value=value,
            confidence=conf if value else 0.0,
            source=source,
            raw_text=chosen.raw if chosen else None,
            readings=[c.reading() for c in cands],
        )
    value = chosen.raw.strip() if chosen and chosen.raw else None
    return Extracted[str](
        value=value,
        confidence=conf if value else 0.0,
        source=source,
        raw_text=chosen.raw if chosen else None,
        bbox=chosen.bbox if chosen else None,  # type: ignore[arg-type]
        readings=[c.reading() for c in cands],
    )


# --------------------------------------------------------------------------- géométrie


def _ring(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(coords) >= 4 and coords[0] == coords[-1]:
        return coords[:-1]
    return coords


def _consistent(
    coords: list[tuple[float, float]],
    area_m2: float | None,
    sides: list[float | None],
    cfg: Settings,
) -> bool:
    """Le jeu de coordonnées est-il cohérent avec la surface et les côtés imprimés ?"""
    ring = _ring(coords)
    if len(ring) < 3:
        return False
    for x, y in ring:
        if not (
            cfg.easting_range[0] <= x <= cfg.easting_range[1]
            and cfg.northing_range[0] <= y <= cfg.northing_range[1]
        ):
            return False
    if not Polygon(ring).is_valid:
        return False
    if area_m2:
        if abs(shoelace_area(ring) - area_m2) / area_m2 * 100 > cfg.area_tolerance_pct:
            return False
    n = len(ring)
    for i, d in enumerate(sides[:n]):
        if (
            d is not None
            and abs(side_length(ring[i], ring[(i + 1) % n]) - d) > cfg.side_tolerance_m
        ):
            return False
    return True


def arbitrate(
    points: list[BoundaryPoint],
    conflicts: list[_Conflict],
    area_m2: float | None,
    sides: list[float | None],
    cfg: Settings,
) -> list[_Candidate] | None:
    """Unique combinaison de lectures qui rend le document cohérent, sinon None.

    Il faut une preuve indépendante des coordonnées : sans surface ni côté imprimés, la
    géométrie ne peut pas trancher."""
    if not conflicts or len(conflicts) > cfg.max_conflicts_for_geometry:
        return None
    if not area_m2 and not any(d is not None for d in sides):
        return None
    base = [[p.x.value, p.y.value] for p in points]
    if any(v is None for xy in base for v in xy):
        return None
    winners = []
    for combo in itertools.product(*(c.options for c in conflicts)):
        coords = [list(xy) for xy in base]
        for conf, cand in zip(conflicts, combo, strict=True):
            coords[conf.point][0 if conf.axis == "x" else 1] = float(cand.normalized)  # type: ignore[arg-type]
        if _consistent([(x, y) for x, y in coords], area_m2, sides, cfg):
            winners.append(list(combo))
            if len(winners) > 1:
                return None
    return winners[0] if winners else None


# --------------------------------------------------------------------------- entrée


def fuse(
    document_id: str,
    paddle: DocumentReading | None,
    qwen: DocumentReading | None,
    cfg: Settings = SETTINGS,
) -> tuple[Extraction, FusionReport]:
    readings = [r for r in (paddle, qwen) if r is not None]
    report = FusionReport()
    pa = paddle.points if paddle else []
    qb = qwen.points if qwen else []
    if pa and qb and len(pa) != len(qb):
        report.issues.append(
            Issue(
                code="POINT_COUNT_MISMATCH",
                severity=Severity.WARNING,
                message=f"PaddleOCR-VL lit {len(pa)} points, Qwen3-VL {len(qb)} : "
                "ligne oubliée ou en trop.",
            )
        )

    points: list[BoundaryPoint] = []
    conflicts: list[_Conflict] = []
    sources = (Source.PADDLE_VL, Source.QWEN_VL)
    for i, pair in enumerate(_align(pa, qb)):
        coords = {}
        for axis in ("x", "y"):
            cands = _coord_cands(pair, axis, sources)
            chosen, src, conf, conflict = _decide(cands, cfg, TABLE_READER,
                                                  cfg.conf_single_coordinate_cap)
            coords[axis] = _extracted_coord(chosen, src, conf, cands)
            if conflict:
                conflicts.append(_Conflict(i, axis, [c for c in cands if c.normalized]))
        label = next((p.label for p in pair if p is not None and p.label), None) or f"P{i + 1}"
        points.append(BoundaryPoint(label=label, row_index=i, x=coords["x"], y=coords["y"]))

    extraction = Extraction(document_id=document_id, points=points)
    for name in ("title_number", "nicad", "owner", "locality"):
        setattr(extraction, name, _fuse_field(readings, name, cfg, report))
    extraction.printed_area = _fuse_field(readings, "area", cfg, report)
    labels = [canonical_label(p.label) for p in points]
    extraction.printed_sides = _fuse_sides(readings, labels, len(_ring_points(points)), cfg)

    report.conflicts_total = len(conflicts)
    if conflicts:
        _resolve(extraction, conflicts, cfg, report)
    return extraction, report


def _ring_points(points: list[BoundaryPoint]) -> list[BoundaryPoint]:
    if (
        len(points) >= 4
        and points[0].complete
        and points[-1].complete
        and (points[0].x.value, points[0].y.value) == (points[-1].x.value, points[-1].y.value)
    ):
        return points[:-1]
    return points


def _resolve(
    extraction: Extraction, conflicts: list[_Conflict], cfg: Settings, report: FusionReport
) -> None:
    pts = extraction.points
    labels = sorted(
        {pts[c.point].label for c in conflicts}, key=lambda lab: [p.label for p in pts].index(lab)
    )
    area = extraction.printed_area.value.m2 if extraction.printed_area.present else None  # type: ignore[union-attr]
    # Seuls les côtés non contestés (une lecture, ou deux identiques) servent de preuve.
    sides = [
        s.value
        if s.present and len({r.normalized for r in s.readings if r.normalized}) == 1
        else None
        for s in extraction.printed_sides
    ]
    choice = arbitrate(pts, conflicts, area, sides, cfg)
    if choice is None:
        detail = "; ".join(
            f"{pts[c.point].label}.{c.axis} : "
            + " / ".join(f"{o.source.value}={o.raw}" for o in c.options)
            for c in conflicts
        )
        report.issues.append(
            Issue(
                code="READING_CONFLICT",
                severity=Severity.WARNING,
                point_labels=labels,
                message=f"Lectures divergentes que la géométrie ne départage pas ({detail}) : "
                "valeur de PaddleOCR-VL gardée, relecture humaine nécessaire.",
            )
        )
        return
    for conf, cand in zip(conflicts, choice, strict=True):
        ex: Extracted[float] = getattr(pts[conf.point], conf.axis)
        ex.value = float(cand.normalized)  # type: ignore[arg-type]
        ex.raw_text = cand.raw
        ex.source = Source.GEOMETRY
        ex.confidence = cfg.conf_geometry
    report.conflicts_resolved = len(conflicts)
    report.issues.append(
        Issue(
            code="READING_CONFLICT_RESOLVED",
            severity=Severity.INFO,
            point_labels=labels,
            message=f"{len(conflicts)} coordonnée(s) lue(s) différemment : "
            "une seule combinaison est "
            "cohérente avec la superficie et les côtés imprimés, elle est retenue.",
        )
    )
