import json
from pathlib import Path

import pytest

from app.core.schema import (
    DATUM_INCONNU,
    DATUM_RRS04,
    UTM_28N,
    Area,
    BoundaryPoint,
    Extracted,
    Extraction,
    Issue,
    Severity,
)
from app.geospatial.parcel import GeometryRefusee, Parcel, parcel

ANNOTATIONS = Path("data/annotations")


def ex(v):
    return Extracted[float](value=v, confidence=1.0)


def extraction(points, aire=None, datum=DATUM_RRS04, issues=()):
    pts = [BoundaryPoint(label=f"B{i+1}", row_index=i, x=ex(x), y=ex(y))
           for i, (x, y) in enumerate(points)]
    return Extraction(
        document_id="D", points=pts, datum=datum, issues=list(issues),
        printed_area=Extracted[Area](value=aire, confidence=1.0) if aire else Extracted[Area](),
    )


CARRE = [(300_000.0, 1_600_000.0), (300_010.0, 1_600_000.0),
         (300_010.0, 1_600_010.0), (300_000.0, 1_600_010.0)]


def test_parcelle_construite():
    p = parcel(extraction(CARRE, Area(ca=100)))
    assert isinstance(p, Parcel)
    assert p.area_m2 == pytest.approx(100.0)
    assert p.printed_m2 == pytest.approx(100.0)
    assert p.crs == UTM_28N and p.datum == DATUM_RRS04 and p.comparable


def test_orientation_independante_du_sens_de_lecture():
    """Un tableau lu dans l'autre sens doit donner la même géométrie."""
    direct = parcel(extraction(CARRE))
    inverse = parcel(extraction(list(reversed(CARRE))))
    assert direct.polygon.equals(inverse.polygon)
    assert direct.area_m2 == pytest.approx(inverse.area_m2)
    assert list(direct.polygon.exterior.coords) == list(inverse.polygon.exterior.coords)


def test_datum_inconnu_rend_la_parcelle_incomparable():
    # La géométrie existe, mais rien en aval ne doit la confronter à une autre source.
    p = parcel(extraction(CARRE, datum=DATUM_INCONNU))
    assert p.area_m2 == pytest.approx(100.0)
    assert not p.comparable


@pytest.mark.parametrize(
    "points, aire, issues, motif",
    [
        ([], None, (), "sommet"),
        (CARRE[:2], None, (), "sommet"),
        (CARRE + [CARRE[0]], None, (), "confondus"),
        # Deux sommets permutés : le polygone se recoupe, on ne le répare pas.
        ([CARRE[0], CARRE[2], CARRE[1], CARRE[3]], None, (), "recoupe"),
        (CARRE, None,
         (Issue(code="COORD_OUT_OF_RANGE", severity=Severity.ERROR, message=""),),
         "corriger d'abord"),
    ],
)
def test_geometrie_refusee(points, aire, issues, motif):
    with pytest.raises(GeometryRefusee, match=motif):
        parcel(extraction(points, aire, issues=issues))


def _annotation(nom):
    a = json.loads((ANNOTATIONS / nom).read_text(encoding="utf-8"))
    aire = a.get("printed_area")
    return extraction([(float(p["x"]), float(p["y"])) for p in a["points"]],
                      Area(**aire) if aire else None)


@pytest.mark.skipif(not ANNOTATIONS.exists(), reason="corpus annoté absent")
@pytest.mark.parametrize("nom, tolerance_pct", [
    ("D10.json", 0.01),   # Thiès, 8 bornes
    ("D11.json", 0.10),   # Mermoz, 4 bornes
    ("D12.json", 0.05),   # Dakar, 12 bornes
    ("D13.json", 0.05),   # Grand Dakar, 4 bornes
    ("D14.json", 0.40),   # Ngaparou, 15 bornes — annotation sous réserve
])
def test_parcelles_reelles(nom, tolerance_pct):
    """Chaque parcelle du corpus retrouve sa superficie imprimée."""
    p = parcel(_annotation(nom))
    assert p.printed_m2
    assert abs(p.area_m2 - p.printed_m2) / p.printed_m2 * 100 <= tolerance_pct
