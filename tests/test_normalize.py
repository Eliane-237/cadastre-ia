import pytest

from model.normalize import (
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
        # Textes réellement lus sur le corpus par Paddle ou Qwen.
        ("05ha 89a 41ca", 58941),
        ("SUPERFICIE : 20° 07 ca", 2007),
        ("44a 89ca", 4489),
        ("80a 05ca", 8005),
        ("Superficie : 02a 21ca", 221),
        ("SURFACE : 21a 00ca", 2100),
        (": 02'55' ca", 255),
        ("Surface = 6a 94ca", 694),
        # L'exposant des ares rendu par une étoile, celui des centiares disparu :
        # les centiares restent à leur place, juste après les ares (images.jfif).
        ("03* 00 00", 300),
        # Plus aucun exposant : la convention ha-a-ca sur deux chiffres suffit (tf.jpg).
        ("02 55", 255),
        ("00 06 49", 649),
        # Un point separe parfois le nombre de son unite (Grand Dakar, lot 31).
        ("Superficie : 8.a 07ca", 807),
    ],
)
def test_superficie(text, m2):
    assert parse_area(text).m2 == m2


@pytest.mark.parametrize(
    "text",
    [
        "lot 5 a",
        None,
        # Trois nombres se disputent la place avant « m² » : illisible, pas 50 m².
        "Lot 59 : 1er 50m²",
        # « 1A » collé à un débris alphanumérique n'est pas un are (im.webp).
        ": 8607E1A 73Ca Surface Ajoutée : 85Ca",
        # Le repli positionnel ne doit jamais fabriquer une surface à partir d'autre chose.
        "Lot 12 59",
        "B1 23",
        "25 09 2024",
        "25/09/24",
        "25-09-24",
        "TF 02 55/DK",
        "Nicad 12 34 56",
        "Date : 16 10 26",
        "page 02 55 suite",
        "02 55 66 77",
        # Algèbre des unités : 100 centiares font un are.
        "02 a 155 ca",
    ],
)
def test_superficie_illisible_refusee(text):
    assert parse_area(text) is None


def test_normalisations_de_champs():
    assert canonical_title("TF n° 12.345/DK") == canonical_title("tf 12345 / dk") == "12345/DK"
    assert canonical_title("Titre sans numéro") == "titre sans numero"
    assert canonical_nicad("0123 4567-89") == "0123456789"
    assert canonical_text("  Awa  NDIAYE ") == canonical_text("awa ndiaye")
    assert canonical_text("Médina") == "medina"
    assert canonical_label("B 01") == canonical_label("b-1") == "B1"


def test_area_model_unchanged():
    assert Area(ha=3, a=12, ca=49).m2 == 31_249
