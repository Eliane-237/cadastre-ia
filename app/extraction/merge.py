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
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import combinations

from shapely.geometry import Polygon

from app.core.config import SETTINGS, Settings
from app.core.normalize import (
    canonical_coordinate,
    canonical_label,
    canonical_length,
    canonical_nicad,
    canonical_text,
    canonical_title,
    parse_area,
)
from app.core.schema import (
    CHAMPS_LOCALITE,
    ETIQUETTES,
    TEXT_FIELDS,
    Area,
    BoundaryPoint,
    Extracted,
    Extraction,
    Issue,
    Reading,
    Severity,
    Source,
)
from app.extraction.reading import DocumentReading, FieldReading, PointReading, SideReading
from app.geospatial.measures import shoelace_area, side_length

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


_NON_DIGIT = re.compile(r"\D")


def _concordant(a: str, b: str, cfg: Settings) -> bool:
    """Deux lectures disent-elles la même chose, à l'orthographe près ?

    Les chiffres doivent être identiques et dans le même ordre. C'est cette
    condition qui empêche d'accepter « 234767.39 » pour « 234787.39 », dont la
    similarité textuelle atteint pourtant 0.89 — au même niveau que des variantes
    d'orthographe légitimes. Seules les lettres et la ponctuation peuvent différer.
    """
    if _NON_DIGIT.sub("", a) != _NON_DIGIT.sub("", b):
        return False
    return SequenceMatcher(None, a, b).ratio() >= cfg.near_agree_ratio


def _ponctuation_parasite(s: str) -> int:
    """Points et virgules qui coupent un mot au lieu de separer des nombres.

    Quand deux lecteurs portent les memes chiffres, ils ont lu la meme valeur : ne reste
    qu'a choisir la forme. « 1.4.DEC.2020 » en compte deux, « 14 DEC 2020 » aucune, et le
    point decimal de « 234773.60 » n'en est pas une — il separe deux chiffres, c'est son
    role. Departager la-dessus plutot qu'a la confiance declaree evite de laisser le
    lecteur le plus sur de lui imposer sa forme abimee.
    """
    n = 0
    for i, c in enumerate(s):
        if c not in ".,":
            continue
        avant = s[i - 1] if i else ""
        apres = s[i + 1] if i + 1 < len(s) else ""
        if avant.isalpha() or apres.isalpha():
            n += 1
    return n


def _mieux_forme(cands: list[_Candidate]) -> _Candidate:
    """La lecture la moins abimee ; a egalite, la plus confiante."""
    return min(
        cands,
        key=lambda c: (_ponctuation_parasite(c.raw or ""), -(c.confidence or 0.0)),
    )


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
    # Lectures qui ne diffèrent que par l'orthographe : accord, à confiance réduite.
    if all(_concordant(a.normalized, b.normalized, cfg) for a, b in combinations(usable, 2)):
        return _mieux_forme(usable), Source.CONSENSUS, cfg.conf_near_agree, False
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
    "mother_title": canonical_title,
    "lot": canonical_title,
    "owner": canonical_text,
    "subdivision": canonical_text,
    "requisition": canonical_text,
    "land_registry": canonical_text,
    "region": canonical_text,
    "department": canonical_text,
    "arrondissement": canonical_text,
    "commune": canonical_text,
    "coordinate_system": canonical_text,
    "scale": canonical_text,
    "certified_on": canonical_text,
    "disclaimer": canonical_text,
    "nicad": canonical_nicad,
}


def _field_cands(readings: list[DocumentReading], name: str) -> list[_Candidate]:
    out = []
    for r in readings:
        f: FieldReading = r.field(name)
        if f.raw is None:
            continue
        if name == "area":
            # Le texte est la recopie littérale du document ; les nombres sont une
            # interprétation du lecteur, qui range parfois les ares dans les hectares
            # (« 80a 05ca » lu ha=80). Le texte fait foi, la structure sert de secours.
            area = parse_area(f.raw) or f.area
            norm = str(int(area.m2)) if area else None
        else:
            norm = _FIELD_NORMALIZERS[name](f.raw)
        out.append(_Candidate(r.source, f.raw, norm, f.confidence, f.bbox))
    return out


def _sans_etiquette(cands: list[_Candidate], name: str) -> list[_Candidate]:
    """Écarte les lectures qui ont rendu l'étiquette de la case au lieu de son contenu.

    « Requérant » dans la case Requérant n'est pas un nom de propriétaire : une case ne
    se nomme pas elle-même. Ce n'est pas un désaccord à arbitrer, c'est une lecture qui
    n'en est pas une. On ne retire jamais la dernière : sans rien en face, mieux vaut
    une valeur douteuse que pas de valeur du tout.
    """
    libelles = ETIQUETTES.get(name, ())
    if not libelles:
        return cands
    garde = [c for c in cands
             if (canonical_text(c.raw) or "") not in libelles]
    return garde if garde else cands


def _tranche_par_repertoire(
    cands: list[_Candidate], name: str, connu: Callable[[str], bool] | None
) -> list[_Candidate]:
    """Pour un champ de localité, une localité connue l'emporte sur une inconnue.

    Règle à SENS UNIQUE, et c'est essentiel : le répertoire ne couvre qu'une partie des
    localités sénégalaises. Une lecture connue bat une inconnue, mais une valeur absente
    du répertoire n'est jamais écartée pour autant — « Mermoz : Sacré Coeur » est réel
    et n'y figure pas. Quand aucune ou plusieurs lectures sont connues, on se tait.
    """
    if connu is None or name not in CHAMPS_LOCALITE:
        return cands
    reconnues = [c for c in cands if c.raw and connu(c.raw)]
    return reconnues if len(reconnues) == 1 else cands


def _fuse_field(
    readings: list[DocumentReading], name: str, cfg: Settings, report: FusionReport,
    connu: Callable[[str], bool] | None = None,
) -> Extracted:
    cands = _field_cands(readings, name)
    # Deux preuves indépendantes des lectures elles-mêmes, dans cet ordre : une étiquette
    # n'est jamais une valeur, puis le répertoire départage ce qui reste.
    if len({c.normalized for c in cands if c.normalized}) > 1:
        cands = _tranche_par_repertoire(_sans_etiquette(cands, name), name, connu)
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
        value = None
        if chosen is not None:
            value = parse_area(chosen.raw)
            structured = next(
                (
                    r.field("area").area
                    for r in readings
                    if r.field("area").raw == chosen.raw and r.field("area").area is not None
                ),
                None,
            )
            if value is None:
                value = structured
            elif structured is not None and structured.m2 != value.m2:
                report.issues.append(
                    Issue(
                        code="AREA_READING_DIVERGENT",
                        severity=Severity.WARNING,
                        message=(
                            f"superficie : le texte « {chosen.raw} » vaut {value.m2:.0f} m², "
                            f"les nombres du lecteur {structured.m2:.0f} m² ; "
                            "le texte est retenu."
                        ),
                    )
                )
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


def _pose(base: list[list[float]], conf: _Conflict, cand: _Candidate) -> list[tuple[float, float]]:
    """Le jeu de coordonnées où ce seul désaccord prend cette valeur-là."""
    coords = [list(xy) for xy in base]
    coords[conf.point][0 if conf.axis == "x" else 1] = float(cand.normalized)  # type: ignore[arg-type]
    return [(x, y) for x, y in coords]


def _elimine_un_a_un(
    base: list[list[float]],
    conflicts: list[_Conflict],
    area_m2: float | None,
    sides: list[float | None],
    cfg: Settings,
) -> dict[int, _Candidate]:
    """Les désaccords qu'une lecture seule suffit à trancher.

    Une valeur qui, posée seule dans un document par ailleurs cohérent, le rend
    incohérent, est fausse : la superficie imprimée la réfute. C'est la même preuve que
    l'énumération complète, mais appliquée à un désaccord à la fois — donc linéaire, et
    surtout capable de trancher les cas évidents sans être paralysée par les autres.

    L'énumération exige en effet UNE combinaison cohérente. Quand deux lectures ne
    diffèrent que d'un centimètre, les deux restent cohérentes, plusieurs combinaisons
    gagnent, et elle abandonne — en emportant au passage les désaccords de 900 m qu'elle
    savait pourtant trancher.

    On recommence tant que des valeurs tombent : une coordonnée corrigée affermit la base
    et peut départager celles qui résistaient.
    """
    tranches: dict[int, _Candidate] = {}
    courant = [list(xy) for xy in base]
    while True:
        trouve = False
        for i, conf in enumerate(conflicts):
            if i in tranches:
                continue
            debout = [c for c in conf.options
                      if _consistent(_pose(courant, conf, c), area_m2, sides, cfg)]
            # Aucune option tenable : le document est faux ailleurs, on ne tranche pas.
            if len(debout) != 1:
                continue
            tranches[i] = debout[0]
            courant[conf.point][0 if conf.axis == "x" else 1] = float(debout[0].normalized)  # type: ignore[arg-type]
            trouve = True
        if not trouve:
            return tranches


def arbitrate(
    points: list[BoundaryPoint],
    conflicts: list[_Conflict],
    area_m2: float | None,
    sides: list[float | None],
    cfg: Settings,
) -> dict[int, _Candidate]:
    """Les désaccords que la géométrie tranche, indexés comme `conflicts`.

    Deux passes. La première élimine un à un ce qui se réfute seul, et c'est elle qui
    fait le gros du travail. La seconde énumère ce qui reste — forcément peu — pour
    attraper les désaccords qui se compensent mutuellement, que seule une vue d'ensemble
    révèle.

    Il faut une preuve indépendante des coordonnées : sans surface ni côté imprimés, la
    géométrie ne peut rien trancher.
    """
    if not conflicts:
        return {}
    if not area_m2 and not any(d is not None for d in sides):
        return {}
    base = [[p.x.value, p.y.value] for p in points]
    if any(v is None for xy in base for v in xy):
        return {}

    tranches = _elimine_un_a_un(base, conflicts, area_m2, sides, cfg)
    for i, cand in tranches.items():
        conf = conflicts[i]
        base[conf.point][0 if conf.axis == "x" else 1] = float(cand.normalized)  # type: ignore[arg-type]

    restants = [(i, c) for i, c in enumerate(conflicts) if i not in tranches]
    if not restants or len(restants) > cfg.max_conflicts_for_geometry:
        return tranches

    winners = []
    for combo in itertools.product(*(c.options for _, c in restants)):
        coords = [list(xy) for xy in base]
        for (_, conf), cand in zip(restants, combo, strict=True):
            coords[conf.point][0 if conf.axis == "x" else 1] = float(cand.normalized)  # type: ignore[arg-type]
        if _consistent([(x, y) for x, y in coords], area_m2, sides, cfg):
            winners.append(list(combo))
            if len(winners) > 1:
                return tranches
    if winners:
        tranches.update({i: cand for (i, _), cand in zip(restants, winners[0], strict=True)})
    return tranches


# --------------------------------------------------------------------------- entrée


def fuse(
    document_id: str,
    paddle: DocumentReading | None,
    qwen: DocumentReading | None,
    cfg: Settings = SETTINGS,
    connu: Callable[[str], bool] | None = None,
) -> tuple[Extraction, FusionReport]:
    """Les deux lectures en une extraction. `connu` dit si un nom est une localité
    répertoriée ; l'appelant le fournit, pour que la fusion n'ait pas à connaître la
    validation."""
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
    for name in TEXT_FIELDS:
        setattr(extraction, name, _fuse_field(readings, name, cfg, report, connu))
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
    area = extraction.printed_area.value.m2 if extraction.printed_area.present else None  # type: ignore[union-attr]
    # Seuls les côtés non contestés (une lecture, ou deux identiques) servent de preuve.
    sides = [
        s.value
        if s.present and len({r.normalized for r in s.readings if r.normalized}) == 1
        else None
        for s in extraction.printed_sides
    ]
    tranches = arbitrate(pts, conflicts, area, sides, cfg)
    for i, cand in tranches.items():
        conf = conflicts[i]
        ex: Extracted[float] = getattr(pts[conf.point], conf.axis)
        ex.value = float(cand.normalized)  # type: ignore[arg-type]
        ex.raw_text = cand.raw
        ex.source = Source.GEOMETRY
        ex.confidence = cfg.conf_geometry
    report.conflicts_resolved = len(tranches)

    def _etiquettes(choisis: list[_Conflict]) -> list[str]:
        ordre = [p.label for p in pts]
        return sorted({pts[c.point].label for c in choisis}, key=ordre.index)

    if tranches:
        report.issues.append(
            Issue(
                code="READING_CONFLICT_RESOLVED",
                severity=Severity.INFO,
                point_labels=_etiquettes([conflicts[i] for i in tranches]),
                message=f"{len(tranches)} coordonnée(s) sur {len(conflicts)} lue(s) "
                "différemment : la superficie et les côtés imprimés désignent une seule "
                "valeur tenable, elle est retenue.",
            )
        )

    restants = [c for i, c in enumerate(conflicts) if i not in tranches]
    if not restants:
        return
    # On ne signale que ce qui reste vraiment en suspens. Noyer les quelques désaccords
    # qui demandent un œil humain au milieu de ceux que la géométrie a tranchés produit
    # une liste qu'on finit par ne plus lire.
    detail = "; ".join(
        f"{pts[c.point].label}.{c.axis} : "
        + " / ".join(f"{o.source.value}={o.raw}" for o in c.options)
        for c in restants
    )
    report.issues.append(
        Issue(
            code="READING_CONFLICT",
            severity=Severity.WARNING,
            point_labels=_etiquettes(restants),
            message=f"{len(restants)} lecture(s) divergente(s) que la géométrie ne "
            f"départage pas ({detail}) : valeur de PaddleOCR-VL gardée, relecture "
            "humaine nécessaire.",
        )
    )
