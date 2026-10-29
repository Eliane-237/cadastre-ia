"""La reference est une parcelle reelle du corpus ; le « plan » est cette meme
parcelle mise a l'echelle, tournee, deplacee, et lue avec le bruit d'un pointage
d'etiquette au pixel pres."""

import math

import numpy as np
import pytest

from app.geospatial.procrustes import RecalageImpossible, concordent, superpose
from tests.fake_readers import sommets


@pytest.fixture
def dakar():
    return sommets("dakar")           # 12 bornes, lues par les modèles


def plan(points, *, echelle, angle_deg, dx=0.0, dy=0.0, bruit_relatif=0.0,
         miroir=False, graine=7):
    """La parcelle telle qu'elle serait dessinee sur le plan, puis pointee."""
    alea = np.random.default_rng(graine)
    t = math.radians(angle_deg)
    rotation = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    p = np.asarray(points, dtype=float)
    p = p - p.mean(axis=0)
    if miroir:
        p[:, 0] *= -1
    taille = np.linalg.norm(p) / math.sqrt(len(p))
    p = (p @ rotation.T) / echelle + np.array([dx, dy])
    return (p + alea.normal(0, bruit_relatif * taille / echelle, p.shape)).tolist()


@pytest.mark.parametrize("echelle, angle", [(1, 0), (500, -17), (5000, 30), (200, 180)])
def test_echelle_et_rotation_retrouvees(dakar, echelle, angle):
    r = superpose(dakar, plan(dakar, echelle=echelle, angle_deg=angle, dx=120, dy=-80))
    assert r.echelle == pytest.approx(echelle, rel=1e-6)
    assert math.cos(math.radians(r.rotation_deg - angle)) == pytest.approx(1.0, abs=1e-9)
    assert r.residu_pct < 1e-6
    assert concordent(r)


def test_pointage_au_pixel_pres(dakar):
    """Une etiquette pointee a 0,3 % de la taille du dessin reste concordante."""
    r = superpose(dakar, plan(dakar, echelle=500, angle_deg=30, bruit_relatif=0.003))
    assert concordent(r, tolerance_pct=1.0)
    assert r.echelle == pytest.approx(500, rel=0.02)


@pytest.mark.parametrize("ecart_m, attendu", [(0.05, True), (0.50, True), (2.00, False)])
def test_un_sommet_deplace(dakar, ecart_m, attendu):
    dessin = [list(p) for p in plan(dakar, echelle=500, angle_deg=30)]
    dessin[4][0] += ecart_m / 500          # le meme ecart, a l'echelle du plan
    assert concordent(superpose(dakar, dessin)) is attendu


def test_parcelle_retournee_detectee(dakar):
    """Un terrain et son miroir ont meme aire et memes cotes : les confondre
    validerait une geometrie retournee."""
    r = superpose(dakar, plan(dakar, echelle=500, angle_deg=30, miroir=True))
    assert r.reflexion and not concordent(r)
    assert "retournement" in r.message


def test_autre_parcelle_rejetee(dakar):
    """Deux terrains differents, pointes pareil, ne doivent pas se superposer."""
    douze = sommets("nguekhokh")[:12]
    r = superpose(dakar, douze)
    assert r.residu_pct > 10 and not concordent(r)


@pytest.mark.parametrize("a, b, motif", [
    ([(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0)], "rien à superposer"),
    ([(0, 0), (1, 0)], [(0, 0), (1, 0)], "au moins"),
    ([(0, 0)] * 3, [(0, 0), (1, 0), (1, 1)], "confondus"),
])
def test_recalage_impossible(a, b, motif):
    with pytest.raises(RecalageImpossible, match=motif):
        superpose(a, b)
