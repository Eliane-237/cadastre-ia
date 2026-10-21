"""Les cas sont construits a partir de parcelles reelles du corpus, deplacees :
les formes restent vraies, seule leur position change."""

import json
from pathlib import Path

import pytest
from shapely.affinity import translate

from app.geospatial.parcel import Parcel, parcel
from app.geospatial.overlap import NonComparable, Verdict, compare, scan
from app.core.schema import (
    DATUM_INCONNU,
    DATUM_RRS04,
    Area,
    BoundaryPoint,
    Extracted,
    Extraction,
)

ANNOTATIONS = Path("data/annotations")
pytestmark = pytest.mark.skipif(not ANNOTATIONS.exists(), reason="corpus annoté absent")


def reelle(nom):
    a = json.loads((ANNOTATIONS / f"{nom}.json").read_text(encoding="utf-8"))
    pts = [BoundaryPoint(label=p["label"], row_index=i,
                         x=Extracted[float](value=float(p["x"]), confidence=1.0),
                         y=Extracted[float](value=float(p["y"]), confidence=1.0))
           for i, p in enumerate(a["points"])]
    aire = a.get("printed_area")
    return parcel(Extraction(
        document_id=a["document_id"], points=pts, datum=DATUM_RRS04,
        printed_area=Extracted[Area](value=Area(**aire), confidence=1.0)
        if aire else Extracted[Area](),
        title_number=Extracted[str](value=a.get("title_number") or a["document_id"],
                                    confidence=1.0)))


def decalee(p, dx, dy, doc, mere=None):
    return Parcel(document_id=doc, polygon=translate(p.polygon, dx, dy), crs=p.crs,
                  datum=p.datum, title_number=doc, mother_title=mere,
                  printed_m2=p.printed_m2)


@pytest.fixture
def mermoz():
    return reelle("D11")          # 255 m², Dakar


@pytest.fixture
def thies():
    return reelle("D10")          # 58 941 m², Thiès


def test_parcelles_eloignees(mermoz):
    assert compare(mermoz, decalee(mermoz, 5000, 5000, "LOIN")) is None


def test_voisins_partageant_une_limite(mermoz):
    # Deux leves de la meme limite different de quelques centimetres : pas un conflit.
    r = compare(mermoz, decalee(mermoz, 13.79, -9.37, "VOISIN"))
    assert r.verdict is Verdict.VOISINAGE and not r.conflit


def test_recouvrement_partiel(mermoz):
    r = compare(mermoz, decalee(mermoz, 5.0, -3.0, "MOITIE"))
    assert r.verdict is Verdict.CHEVAUCHEMENT and r.conflit
    assert 50 < r.part_pct < 80


def test_titre_entierement_revendu(mermoz):
    r = compare(mermoz, decalee(mermoz, 0, 0, "DOUBLE"))
    assert r.verdict is Verdict.INCLUSION and r.conflit
    assert r.part_pct == pytest.approx(100.0)


def test_morcellement_explique_par_le_titre_mere(mermoz, thies):
    """Une parcelle detachee d'un titre mere n'est pas une fraude : le document de
    Thies porte trois fois « Surplus TF 6652/TH »."""
    dedans = decalee(mermoz, 294_900 - 234_773, 1_640_800 - 1_627_666, "PLLE")
    assert compare(dedans, thies).verdict is Verdict.INCLUSION

    legitime = Parcel(**{**dedans.__dict__, "mother_title": thies.title_number})
    r = compare(legitime, thies)
    assert r.verdict is Verdict.MORCELLEMENT and not r.conflit


@pytest.mark.parametrize("datum", [DATUM_INCONNU, "WGS84"])
def test_referentiel_incompatible_refuse(mermoz, thies, datum):
    autre = Parcel(**{**mermoz.__dict__, "datum": datum, "document_id": "X"})
    with pytest.raises(NonComparable):
        compare(autre, thies)


def test_balayage_conflits_en_tete(mermoz, thies):
    base = [thies,
            decalee(mermoz, 5000, 5000, "LOIN"),
            decalee(mermoz, 13.79, -9.37, "VOISIN"),
            decalee(mermoz, 5.0, -3.0, "MOITIE")]
    trouves = scan(mermoz, base)
    assert [r.verdict for r in trouves] == [Verdict.CHEVAUCHEMENT, Verdict.VOISINAGE]
    assert trouves[0].conflit
