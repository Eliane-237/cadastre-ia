import pytest

from model.numbers import (
    canonical_coordinate,
    canonical_label,
    canonical_length,
    canonical_nicad,
    canonical_number,
    canonical_text,
    canonical_title,
    parse_area,
)
from model.schema import Area


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("381210.57", "381210.57"),
        ("381 210,57", "381210.57"),
        ("1 564 042,96", "1564042.96"),
        ("1.564.042,96", "1564042.96"),
        ("1,564,042.96", "1564042.96"),
        ("235906.620", "235906.620"),  # zéro final conservé
        ("38121O.57", "381210.57"),  # O lu à la place de 0
        ("1564042", "1564042"),
        ("abc", None),
        (None, None),
    ],
)
def test_canonical_number(raw, expected):
    assert canonical_number(raw) == expected


def test_coordonnee_et_longueur():
    assert canonical_coordinate("25.30") is None
    assert canonical_length("25,30 m") == "25.30"
    assert canonical_length("381210.57") is None


@pytest.mark.parametrize(
    "text, m2",
    [
        ("00 ha 06 a 49 ca", 649),
        ("06 a 49 ca", 649),
        ("1 ha 20 a 35 ca", 12035),
        ("01a63 ca", 163),
        ("649 m²", 649),
        ("283 m?", 283),
        ("12 ha", 120000),
    ],
)
def test_superficie(text, m2):
    assert parse_area(text).m2 == m2


def test_superficie_ambigue_refusee():
    assert parse_area("lot 5 a") is None
    assert parse_area(None) is None


def test_normalisations_de_champs():
    assert canonical_title("TF n° 12.345/DK") == canonical_title("tf 12345 / dk") == "12345/DK"
    assert canonical_title("Titre sans numéro") == "titre sans numero"
    assert canonical_nicad("0123 4567-89") == "0123456789"
    assert canonical_text("  Awa  NDIAYE ") == canonical_text("awa ndiaye")
    assert canonical_text("Médina") == "medina"
    assert canonical_label("B 01") == canonical_label("b-1") == "B1"


def test_area_model_unchanged():
    assert Area(ha=3, a=12, ca=49).m2 == 31_249
