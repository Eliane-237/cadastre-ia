"""Validation géométrique d'une extraction. Fonctions pures : aucune image, aucun OCR.

Le résultat est une liste d'`Issue` : des alertes d'aide à la décision. Un écart de superficie
peut venir d'une mauvaise lecture, d'une erreur dans le document d'origine, ou d'autre chose :
ce module constate, il ne conclut pas à la fraude.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter

from shapely.geometry import Polygon

from model.config import SETTINGS, Settings
from model.normalize import canonical_label
from model.schema import (
    DATUM_INCONNU,
    DATUM_RRS04,
    BoundaryPoint,
    Extraction,
    Issue,
    Severity,
)

Coord = tuple[float, float]


def shoelace_area(coords: list[Coord]) -> float:
    """Superficie par la formule de Gauss. Coordonnées recentrées sur le premier point pour
    éviter la perte de précision (les valeurs UTM font ~1e6 et leurs produits ~1e12)."""
    x0, y0 = coords[0]
    pts = [(x - x0, y - y0) for x, y in coords]
    s = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
            for i in range(len(pts)))
    return abs(s) / 2


def side_length(a: Coord, b: Coord) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _coords(points: list[BoundaryPoint]) -> list[Coord]:
    return [(p.x.value, p.y.value) for p in points]  # type: ignore[misc]  # complétude vérifiée avant


def _drop_closing_point(points: list[BoundaryPoint]) -> list[BoundaryPoint]:
    """Certains tableaux répètent le premier point à la fin pour fermer le polygone."""
    if len(points) >= 4 and points[0].complete and points[-1].complete \
            and points[0].x.value == points[-1].x.value and points[0].y.value == points[-1].y.value:
        return points[:-1]
    return points


# Systemes de reference rencontres sur les titres senegalais. RRS / RSS sont le reseau
# national, exprime en UTM 28N ; HATT est une autre projection, incompatible avec nos
# plages et avec un calcul de surface en metres.
_CRS_UTM28N = ("RRS", "RSS")
_CRS_AUTRES = ("HATT",)


def crs_is_utm28n(declared: str | None) -> bool | None:
    """True si le systeme declare est compatible UTM 28N, False si non, None si inconnu."""
    if not declared:
        return None
    text = declared.upper()
    if any(k in text for k in _CRS_AUTRES):
        return False
    if any(k in text for k in _CRS_UTM28N):
        return True
    return None


def datum_for(declared: str | None) -> str:
    """Le datum derrière le système imprimé sur le document, ou « inconnu ».

    « RRS » et « RSS » désignent le Réseau de Référence du Sénégal, dont le datum est
    documenté. Tout le reste reste inconnu : il vaut mieux qu'un traitement en aval
    bute sur un datum manquant plutôt qu'il suppose du WGS84 sans le savoir.
    """
    return DATUM_RRS04 if crs_is_utm28n(declared) is True else DATUM_INCONNU


_NUMERO = re.compile(r"([A-Z]*)(\d+)")


def _check_labels(points: list[BoundaryPoint]) -> list[Issue]:
    """Les étiquettes des sommets doivent être uniques et former une suite sans trou.

    Une étiquette répétée trahit une ligne du tableau recopiée par l'OCR. Un trou dans
    la suite — B1, B2, B4 — trahit une ligne perdue : le polygone se referme alors sur
    un sommet de moins, sans rien devenir invalide, et la surface calculée est fausse
    sans que rien ne proteste.
    """
    issues: list[Issue] = []
    canon = [canonical_label(p.label) for p in points]

    for lab, n in Counter(c for c in canon if c).items():
        if n > 1:
            issues.append(
                Issue(
                    code="LABEL_DUPLICATE",
                    severity=Severity.ERROR,
                    point_labels=[lab],
                    message=f"L'étiquette « {lab} » apparaît {n} fois : une ligne du "
                            "tableau a été lue deux fois.",
                )
            )

    parts = [m.groups() for c in canon if c and (m := _NUMERO.fullmatch(c))]
    if not parts or len(parts) != len(points):   # étiquettes absentes ou non numérotées
        return issues

    prefixes = {p for p, _ in parts}
    if len(prefixes) > 1:
        issues.append(
            Issue(
                code="LABEL_PREFIX_MIXTE",
                severity=Severity.WARNING,
                message="Les étiquettes mélangent les préfixes "
                        f"({', '.join(sorted(repr(p) for p in prefixes))}) : "
                        "l'une d'elles est probablement mal lue.",
            )
        )

    numeros = sorted({int(n) for _, n in parts})
    manquants = [n for n in range(numeros[0], numeros[-1] + 1) if n not in set(numeros)]
    if manquants:
        issues.append(
            Issue(
                code="LABEL_SEQUENCE",
                severity=Severity.WARNING,
                message=f"La suite des sommets saute {', '.join(map(str, manquants))} : "
                        "une ligne du tableau manque peut-être.",
            )
        )
    return issues


def _check_duplicates(points: list[BoundaryPoint], cfg: Settings) -> list[Issue]:
    """Deux sommets distincts ne peuvent pas occuper le même point.

    Le point de fermeture légitime — première ligne répétée en fin de tableau — a déjà
    été retiré en amont. Ce qui reste est une recopie accidentelle.
    """
    issues: list[Issue] = []
    complets = [p for p in points if p.complete]

    positions: dict[tuple[float, float], list[str]] = {}
    for p in complets:
        positions.setdefault((p.x.value, p.y.value), []).append(p.label)
    for (x, y), labels in positions.items():
        if len(labels) > 1:
            issues.append(
                Issue(
                    code="POINT_DUPLICATE",
                    severity=Severity.ERROR,
                    point_labels=labels,
                    message=f"{' et '.join(labels)} ont les mêmes coordonnées "
                            f"({x:.2f}, {y:.2f}) : une ligne a été recopiée.",
                )
            )

    # Une même valeur sur trois lignes ou plus d'une colonne : la case a été recopiée.
    # Deux sommets peuvent légitimement partager une abscisse (parcelle rectangulaire) ;
    # trois, c'est que trois bornes seraient alignées, ce qui n'arrive pour ainsi dire pas.
    for axe, valeurs in (("X", [p.x.value for p in complets]), ("Y", [p.y.value for p in complets])):
        for valeur, n in Counter(valeurs).items():
            if n >= cfg.max_identical_coordinates:
                issues.append(
                    Issue(
                        code="COORD_COLUMN_SUSPECT",
                        severity=Severity.WARNING,
                        point_labels=[p.label for p in complets
                                      if (p.x.value if axe == "X" else p.y.value) == valeur],
                        message=f"La valeur {valeur:.2f} apparaît {n} fois en {axe} : "
                                "la case a probablement été recopiée d'une ligne à l'autre.",
                    )
                )
    return issues


def _out_of_range(p: BoundaryPoint, cfg: Settings) -> list[str]:
    bad = []
    if not cfg.easting_range[0] <= p.x.value <= cfg.easting_range[1]:  # type: ignore[operator]
        bad.append(f"X={p.x.value}")
    if not cfg.northing_range[0] <= p.y.value <= cfg.northing_range[1]:  # type: ignore[operator]
        bad.append(f"Y={p.y.value}")
    return bad


def _check_values(points: list[BoundaryPoint], cfg: Settings) -> list[Issue]:
    issues: list[Issue] = []

    # Si la plupart des points sortent de la plage, ce n'est pas une lecture fautive :
    # le document est dans un autre referentiel. Un seul point hors plage, en revanche,
    # est bien un chiffre mal lu (« 38210.57 » pour « 381210.57 »).
    complete = [p for p in points if p.complete]
    hors = [p for p in complete if _out_of_range(p, cfg)]
    if complete and len(hors) > len(complete) / 2:
        return [
            Issue(
                code="CRS_HORS_UTM28N",
                severity=Severity.WARNING,
                point_labels=[p.label for p in hors],
                message=(
                    f"{len(hors)} des {len(complete)} point(s) complets hors de la plage "
                    f"UTM 28N (sur {len(points)} lus) : le document est probablement "
                    "exprimé dans un autre système de coordonnées. Contrôles géométriques "
                    "non applicables."
                ),
            )
        ]

    for p in points:
        if not p.complete:
            missing = [a for a in ("x", "y") if not getattr(p, a).present]
            issues.append(Issue(code="POINT_INCOMPLETE", severity=Severity.ERROR, point_labels=[p.label],
                                message=f"Point {p.label} : coordonnée {', '.join(missing)} non lue."))
            continue
        bad = _out_of_range(p, cfg)
        if bad:
            issues.append(Issue(code="COORD_OUT_OF_RANGE", severity=Severity.ERROR, point_labels=[p.label],
                                message=f"Point {p.label} hors de la plage UTM 28N attendue ({', '.join(bad)})."))
    ok = [p for p in points if p.complete]
    if len(ok) >= 3:
        mx = statistics.median(p.x.value for p in ok)  # type: ignore[misc]
        my = statistics.median(p.y.value for p in ok)  # type: ignore[misc]
        flagged = {lab for i in issues for lab in i.point_labels}
        for p in ok:
            if p.label in flagged:
                continue
            if max(abs(p.x.value - mx), abs(p.y.value - my)) > cfg.max_parcel_extent_m:  # type: ignore[operator]
                issues.append(Issue(code="COORD_OUTLIER", severity=Severity.ERROR, point_labels=[p.label],
                                    message=f"Point {p.label} à plus de {cfg.max_parcel_extent_m:.0f} m "
                                            "des autres : probablement mal lu."))
    return issues


def _regular_polygon_area(side: float, n: int) -> float:
    """Aire du polygone régulier : le maximum atteignable avec n côtés tous égaux."""
    return n * side * side / (4 * math.tan(math.pi / n))


def unreliable_sides(
    printed: list[float | None], printed_m2: float | None, cfg: Settings = SETTINGS
) -> str | None:
    """Pourquoi la colonne des côtés imprimés n'est pas croyable, ou None si elle l'est.

    Cette colonne sert de preuve indépendante pour départager deux lectures d'une
    coordonnée. Mal lue, elle fait condamner des lectures correctes et absoudre des
    fausses : une colonne absente ne coûte qu'une contrainte en moins, une colonne
    fausse coûte la vérité. En cas de doute on la rejette entière, car une colonne
    où une case a mal été découpée n'est fiable sur aucune de ses lignes.
    """
    reels = [c for c in printed if c is not None]
    if not reels:
        return None
    perimetre = sum(reels)

    # Inégalité polygonale : dans tout polygone fermé, un côté est strictement
    # inférieur à la somme de tous les autres.
    for c in reels:
        if c >= perimetre - c:
            return f"côté imprimé {c:.2f} m supérieur à la somme des autres ({perimetre - c:.2f} m)"

    if printed_m2:
        caracteristique = math.sqrt(printed_m2)
        debris = [c for c in reels if c < cfg.sides_fragment_ratio * caracteristique]
        if debris:
            return (
                f"côté(s) imprimé(s) {', '.join(f'{c:.2f}' for c in debris)} m sous "
                f"{cfg.sides_fragment_ratio:.0%} de la longueur caractéristique "
                f"({caracteristique:.2f} m) : découpage de cellule manqué"
            )
        if len(set(reels)) == 1 and len(reels) >= 3:
            maxi = _regular_polygon_area(reels[0], len(reels))
            if abs(maxi - printed_m2) / printed_m2 > cfg.sides_regular_tolerance_pct / 100:
                return (
                    f"colonne constante ({reels[0]:.2f} m sur {len(reels)} lignes) "
                    f"incompatible avec la superficie : un polygone régulier de ce côté "
                    f"ferait {maxi:.0f} m² contre {printed_m2:.0f} m² imprimés ; "
                    "la case a été recopiée"
                )
        # Inégalité isopérimétrique : aucune forme fermée de périmètre P n'enferme
        # plus de P²/4π. Ne vaut que si la colonne est complète.
        if len(reels) == len(printed) and printed_m2 > perimetre**2 / (4 * math.pi):
            return (
                f"superficie imprimée ({printed_m2:.0f} m²) supérieure au maximum "
                f"qu'enferme ce périmètre ({perimetre:.2f} m donne au plus "
                f"{perimetre ** 2 / (4 * math.pi):.0f} m²)"
            )
    return None


def _chaine(valeur, brut: str | None) -> str:
    """La coordonnee telle qu'imprimee, ou a defaut telle que lue."""
    return (brut or "").strip() or f"{valeur:.2f}"


def _dominant(valeurs: list, part_min: float):
    """La valeur partagee par au moins `part_min` de la colonne, ou None."""
    if len(valeurs) < 3:
        return None
    valeur, combien = Counter(valeurs).most_common(1)[0]
    return valeur if combien / len(valeurs) >= part_min else None


def _prefixe_dominant(chaines: list[str], part_min: float) -> str | None:
    """Le plus long prefixe de chiffres partage par au moins `part_min` de la colonne.

    Les tableaux cadastraux omettent parfois le prefixe des milliers sur les lignes
    suivantes, par convention d'ecriture. Une valeur qui ne porte pas le prefixe de
    ses voisines est soit une lecture tronquee, soit cette convention : dans les deux
    cas elle demande un oeil.
    """
    entiers = [c.split(".")[0] for c in chaines]
    if len(entiers) < 3:
        return None
    for longueur in range(min(len(e) for e in entiers), 1, -1):
        candidat = _dominant([e[:longueur] for e in entiers], part_min)
        if candidat is not None:
            return candidat
    return None


def _check_columns(points: list[BoundaryPoint], cfg: Settings) -> list[Issue]:
    """Regularite des colonnes X et Y, independamment de toute geometrie.

    Une colonne de coordonnees cadastrales est regulière : meme nombre de chiffres,
    meme nombre de decimales, meme prefixe geographique. Une valeur qui s'en ecarte
    est suspecte. Ces controles ne corrigent rien et n'accusent rien : ils valent
    surtout quand la superficie imprimee manque, c'est-a-dire quand la geometrie n'a
    aucun juge. Ils restent des avertissements, car une erreur de lecture systematique
    rendrait le motif dominant lui-meme faux.
    """
    issues: list[Issue] = []
    for axe, champs in (("X", [(p, p.x) for p in points if p.x.present]),
                        ("Y", [(p, p.y) for p in points if p.y.present])):
        if len(champs) < 3:
            continue
        chaines = [_chaine(c.value, getattr(c, "raw_text", None)) for _, c in champs]

        longueurs = [len(c.split(".")[0]) for c in chaines]
        attendue = _dominant(longueurs, cfg.column_min_share)
        if attendue is not None:
            ecarts = [p.label for (p, _), n in zip(champs, longueurs) if n != attendue]
            if ecarts:
                issues.append(Issue(
                    code="COLUMN_LENGTH", severity=Severity.WARNING, point_labels=ecarts,
                    message=f"{axe} : {', '.join(ecarts)} n'a/n'ont pas {attendue} chiffres "
                            "avant la virgule, contrairement au reste de la colonne.",
                ))

        decimales = [len(c.split(".")[1]) if "." in c else 0 for c in chaines]
        attendues = _dominant(decimales, cfg.column_min_share)
        if attendues is not None:
            ecarts = [p.label for (p, _), n in zip(champs, decimales) if n != attendues]
            if ecarts:
                issues.append(Issue(
                    code="COLUMN_DECIMALS", severity=Severity.WARNING, point_labels=ecarts,
                    message=f"{axe} : {', '.join(ecarts)} n'a/n'ont pas {attendues} décimale(s), "
                            "contrairement au reste de la colonne.",
                ))

        prefixe = _prefixe_dominant(chaines, cfg.column_min_share)
        if prefixe is not None:
            ecarts = [p.label for (p, _), c in zip(champs, chaines)
                      if not c.split(".")[0].startswith(prefixe)]
            if ecarts:
                issues.append(Issue(
                    code="COLUMN_PREFIX", severity=Severity.WARNING, point_labels=ecarts,
                    message=f"{axe} : {', '.join(ecarts)} ne commence(nt) pas par « {prefixe} », "
                            "contrairement au reste de la colonne.",
                ))
    return issues


def _check_sides(
    points: list[BoundaryPoint], printed: list, printed_m2: float | None, cfg: Settings
) -> list[Issue]:
    # Une colonne mal lue produirait un côté fautif par ligne et ferait accuser des
    # coordonnées correctes : on la rejette d'un bloc, avec sa raison.
    if raison := unreliable_sides(
        [d.value if d.present else None for d in printed], printed_m2, cfg
    ):
        return [
            Issue(
                code="PRINTED_SIDES_UNRELIABLE",
                severity=Severity.WARNING,
                message=f"Longueurs imprimées écartées — {raison}.",
            )
        ]
    n = len(points)
    coords = _coords(points)
    failing: list[int] = []
    for i in range(min(n, len(printed))):
        d = printed[i]
        if not d.present:
            continue
        calc = side_length(coords[i], coords[(i + 1) % n])
        if abs(calc - d.value) > cfg.side_tolerance_m:
            failing.append(i)
    issues = [Issue(code="SIDE_MISMATCH", severity=Severity.WARNING,
                    point_labels=[points[i].label, points[(i + 1) % n].label],
                    message=f"Côté {points[i].label}-{points[(i + 1) % n].label} : calculé "
                            f"{side_length(coords[i], coords[(i + 1) % n]):.2f} m, imprimé "
                            f"{printed[i].value:.2f} m.") for i in failing]
    # Un sommet commun à deux côtés fautifs consécutifs est le suspect le plus probable.
    fset = set(failing)
    for i in sorted(fset):
        if (i - 1) % n in fset and n > 2:
            issues.append(Issue(code="POINT_SUSPECT", severity=Severity.WARNING, point_labels=[points[i].label],
                                message=f"Point {points[i].label} : les deux côtés qui y aboutissent "
                                        "sont incohérents avec le plan."))
    return issues


def validate(extraction: Extraction, cfg: Settings = SETTINGS) -> list[Issue]:
    """Contrôles de cohérence. Les contrôles géométriques ne sont pas faits si une coordonnée est
    manquante ou aberrante : calculer sur des valeurs connues pour fausses induirait en erreur."""
    points = _drop_closing_point(extraction.points)
    issues: list[Issue] = []

    declared = extraction.coordinate_system.value if extraction.coordinate_system.present else None
    if crs_is_utm28n(declared) is False:
        issues.append(
            Issue(
                code="CRS_NON_SUPPORTE",
                severity=Severity.WARNING,
                message=(
                    f"Système de coordonnées « {declared} » : ni la plage attendue ni le "
                    "calcul de surface en UTM 28N ne s'appliquent. Géométrie non contrôlée."
                ),
            )
        )
        return issues
    if declared and crs_is_utm28n(declared) is None:
        issues.append(
            Issue(
                code="CRS_INCONNU",
                severity=Severity.WARNING,
                message=(
                    f"Système de coordonnées « {declared} » non reconnu : le datum reste "
                    "inconnu et les coordonnées ne peuvent pas être confrontées à une "
                    "autre source. Vérifier la lecture de ce champ."
                ),
            )
        )

    issues += _check_labels(points)
    issues += _check_duplicates(points, cfg)
    issues += _check_columns(points, cfg)
    issues += _check_values(points, cfg)

    if any(i.severity is Severity.ERROR for i in issues) or any(
        i.code == "CRS_HORS_UTM28N" for i in issues
    ):
        issues.append(Issue(code="GEOMETRY_SKIPPED", severity=Severity.INFO,
                            message="Superficie et côtés non contrôlés : des coordonnées sont à corriger d'abord."))
        return issues
    if len(points) < 3:
        issues.append(Issue(code="TOO_FEW_POINTS", severity=Severity.ERROR,
                            message=f"{len(points)} point(s) lu(s) : au moins 3 sont nécessaires."))
        return issues

    coords = _coords(points)
    if not Polygon(coords).is_valid:
        issues.append(Issue(code="POLYGON_INVALID", severity=Severity.ERROR,
                            point_labels=[p.label for p in points],
                            message="Le polygone se recoupe lui-même : ordre des points ou lecture à vérifier."))

    area = shoelace_area(coords)
    printed_m2: float | None = None
    if extraction.printed_area.present:
        printed = extraction.printed_area.value.m2  # type: ignore[union-attr]
        printed_m2 = float(printed) if printed else None
        gap = abs(area - printed) / printed * 100 if printed else math.inf
        if gap > cfg.area_tolerance_pct:
            issues.append(Issue(code="AREA_MISMATCH", severity=Severity.ERROR,
                                message=f"Superficie calculée {area:.1f} m² contre {printed:.0f} m² imprimés "
                                        f"(écart {gap:.1f} %, tolérance {cfg.area_tolerance_pct} %)."))
    else:
        issues.append(Issue(code="AREA_UNAVAILABLE", severity=Severity.INFO,
                            message="Superficie imprimée non lue : cohérence non vérifiée."))

    issues.extend(_check_sides(points, extraction.printed_sides, printed_m2, cfg))
    return issues


def validated(extraction: Extraction, cfg: Settings = SETTINGS) -> Extraction:
    """Copie de l'extraction avec son datum renseigné et les anomalies détectées ajoutées."""
    declared = extraction.coordinate_system.value if extraction.coordinate_system.present else None
    return extraction.model_copy(
        update={
            "datum": datum_for(declared),
            "issues": [*extraction.issues, *validate(extraction, cfg)],
        }
    )