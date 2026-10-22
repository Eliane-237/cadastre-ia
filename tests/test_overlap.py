"""Les cas sont construits a partir de parcelles reelles du corpus, deplacees :
les formes restent vraies, seule leur position change."""

import json
from pathlib import Path

import pytest
from shapely.affinity import translate

from app.core.schema import (
    DATUM_INCONNU,
    DATUM_RRS04,
    Area,
    BoundaryPoint,
    Extracted,
    Extraction,
)
from app.geospatial.overlap import NonComparable, Verdict, compare, scan
from app.geospatial.parcel import Parcel, parcel

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
    # La bande commune fait 1,8 mm pour 1,4 cm d'incertitude de leve : le verdict le dit
    # en centimetres, au lieu de « moins de 1 % de la parcelle ».
    r = compare(mermoz, decalee(mermoz, 13.79, -9.37, "VOISIN"))
    assert r.verdict is Verdict.SOUS_INCERTITUDE and not r.conflit


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
    assert [r.verdict for r in trouves] == [Verdict.CHEVAUCHEMENT, Verdict.SOUS_INCERTITUDE]
    assert trouves[0].conflit


# --- Précision de position et recouvrement sous incertitude -------------------
# Un recouvrement plus étroit que ce que les deux levés savent distinguer ne doit
# pas être constaté. Avant ces tests, l'aire était rendue sans savoir si elle
# était du signal ou du bruit.

def test_precision_se_deduit_de_l_ecart_de_superficie():
    """ε ≈ |aire calculée − aire imprimée| / périmètre."""
    a = reelle("D11")
    attendu = abs(a.area_m2 - a.printed_m2) / a.polygon.length
    assert a.precision_m == pytest.approx(max(attendu, 0.01))


def test_precision_inconnue_sans_superficie_imprimee():
    """Pas de superficie au document : la qualité se dit inconnue, pas par défaut."""
    sans = Parcel(document_id="X", polygon=reelle("D11").polygon, crs="EPSG:32628",
                  datum=DATUM_RRS04, printed_m2=None)
    assert sans.precision_m is None
    assert sans.qualite == "?"


def test_recouvrement_plus_etroit_que_l_incertitude_n_est_pas_un_conflit():
    """Une bande de 1,8 mm entre deux levés sûrs à 1 cm : il n'y a rien à constater.

    Ce qui compte est la largeur de la bande commune, pas son aire : une parcelle
    décalée d'un centimètre se recouvre à 99,99 %, et ce n'est pas du bruit de levé,
    c'est le même terrain lu deux fois.
    """
    a = reelle("D11")
    b = decalee(a, 13.79, -9.37, "VOISIN")
    r = compare(a, b)
    assert r is not None
    assert r.largeur_m < r.incertitude_m
    assert r.verdict is Verdict.SOUS_INCERTITUDE
    assert r.conflit is False
    assert "rien de mesurable" in r.message


def test_vrai_chevauchement_reste_un_conflit_et_annonce_l_incertitude():
    a = reelle("D11")
    b = decalee(a, 5.0, -3.0, "VOISIN")          # bande de 6,3 m : largement mesurable
    r = compare(a, b)
    assert r is not None
    assert r.conflit is True
    assert r.incertitude_m is not None
    assert "au-delà de l'incertitude" in r.message


def test_incertitude_inconnue_dit_qu_elle_l_est():
    a = reelle("D11")
    b = Parcel(document_id="VOISIN", polygon=translate(a.polygon, 5.0, -3.0),
               crs=a.crs, datum=a.datum, title_number="VOISIN", printed_m2=None)
    r = compare(a, b)
    assert r is not None
    assert r.incertitude_m is None
    assert "sans que la précision des levés soit connue" in r.message
