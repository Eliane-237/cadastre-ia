"""Réparation d'un chiffre sous contrainte géométrique.

Les cas viennent du corpus réel : les valeurs attendues sont celles lues sur les documents
et validées par leur superficie imprimée.
"""

from app.extraction.digit_repair import find_repair, shoelace

# tf.jpg — TITRE N° 22 665/GR, Mermoz / Sacré-Cœur. Superficie imprimée 02a 55ca = 255 m².
TF_X = ["234773.60", "234787.39", "234778.76", "234765.02"]
TF_Y = ["1627666.10", "1627656.73", "1627644.03", "1627653.48"]
TF_AREA = 255.0
TF_SIDES = [16.67, 15.35, 16.68, 15.26]

# tf kaolack — superficie imprimée 02a 21ca = 221 m².
KL_X = ["381210.57", "381225.56", "381219.40", "381204.41"]
KL_Y = ["1564042.96", "1564034.94", "1564023.49", "1564031.51"]
KL_AREA = 221.0


def test_lecture_correcte_nest_pas_touchee():
    assert find_repair(TF_X, TF_Y, TF_AREA, printed_sides=TF_SIDES) is None


def test_substitution_tranchee_par_les_cotes():
    """Les deux lecteurs lisent « 234776.76 » : l'arbitrage entre eux ne peut rien.
    Quatre corrections d'un chiffre donnent la bonne aire ; seule une respecte les cotes."""
    lu = list(TF_X)
    lu[2] = "234776.76"
    repair = find_repair(lu, TF_Y, TF_AREA, printed_sides=TF_SIDES)
    assert repair is not None
    assert (repair.index, repair.axis, repair.after) == (2, "x", "234778.76")
    assert repair.kind == "substitution"


def test_sans_les_cotes_la_surface_seule_ne_tranche_pas():
    """Garde-fou : plusieurs variantes conviennent, donc on ne répare pas."""
    lu = list(TF_X)
    lu[2] = "234776.76"
    assert find_repair(lu, TF_Y, TF_AREA) is None


def test_chiffre_manquant_restitue():
    """« 38210.57 » dans une colonne à six chiffres : un chiffre a été perdu."""
    lu = list(KL_X)
    lu[0] = "38210.57"
    repair = find_repair(lu, KL_Y, KL_AREA)
    assert repair is not None
    assert (repair.index, repair.axis, repair.after) == (0, "x", "381210.57")
    assert repair.kind == "insertion"


def test_deux_erreurs_simultanees_ne_sont_pas_reparees():
    lu_x = list(TF_X)
    lu_x[2] = "234776.76"
    lu_y = list(TF_Y)
    lu_y[3] = "1627653.98"
    assert find_repair(lu_x, lu_y, TF_AREA, printed_sides=TF_SIDES) is None


def test_superficie_absente_ne_declenche_rien():
    lu = list(TF_X)
    lu[2] = "234776.76"
    assert find_repair(lu, TF_Y, 0.0) is None


def test_aire_de_reference():
    points = [(float(x), float(y)) for x, y in zip(TF_X, TF_Y, strict=False)]
    assert abs(shoelace(points) - 255.24) < 0.01
