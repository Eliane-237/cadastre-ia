from pathlib import Path

import pytest
from openpyxl import Workbook
from pydantic import ValidationError

from eval.xlsx_to_annotations import convert
from model.annotation import AnnotatedPoint, Annotation, load_all, save_annotation


def test_coordonnee_mal_formee_refusee():
    with pytest.raises(ValidationError):
        AnnotatedPoint(label="B1", x="12.5", y="1564042.96")


def test_labels_en_double_refuses():
    p = AnnotatedPoint(label="B1", x="381210.57", y="1564042.96")
    with pytest.raises(ValidationError):
        Annotation(document_id="D", image="a.png", points=[p, p])


def test_aller_retour_fichier_conserve_les_decimales(tmp_path: Path):
    a = Annotation(document_id="D01", image="a.png", decimals=3,
                   points=[AnnotatedPoint(label="B1", x="235906.620", y="1624690.747")])
    save_annotation(a, tmp_path)
    (b,) = load_all(tmp_path)
    assert b.points[0].x == "235906.620"  # le zéro final n'est pas perdu
    assert b.verified is False


def _classeur(path: Path) -> None:
    wb = Workbook()
    d = wb.active
    d.title = "Documents"
    d.append(["ID", "Fichier", "Livre", "Commune", "Titre", "NICAD", "ha", "a", "ca", "m2",
              "n", "Décimales", "Échelle", "Date", "Syst", "Qualité", "Statut", "Remarques"])
    d.append(["D01", "a.png", "Dakar", "Médina", "TM 1738/DK", "(vide)", 0, 6, 49, 649,
              2, 3, "", "", "", "Bonne", "Testé", None])
    d.append(["D03", "c.png", "Dakar", "SC2", "TF 1", "non visible", 0, 2, 58, 258,
              None, None, "", "", "", "Faible", "Non testé", None])
    c = wb.create_sheet("Coordonnées")
    c.append(["ID", "Point", "X", "Y"])
    c.append(["D01", "B1", 235906.62, 1624690.747])
    c.append(["D01", "B2", 235911.823, 1624686.99])
    wb.save(path)


def test_conversion_xlsx(tmp_path: Path):
    x = tmp_path / "c.xlsx"
    _classeur(x)
    out = tmp_path / "ann"
    assert [p.name for p in convert(x, out)] == ["D01.json"]   # D03 sans tableau : ignoré
    (a,) = load_all(out)
    assert a.nicad is None and a.title_number == "TM 1738/DK"
    assert a.printed_area.m2 == 649
    assert [p.x for p in a.points] == ["235906.620", "235911.823"]
    assert a.verified is False


def test_chemin_invalide_message_clair(tmp_path: Path):
    with pytest.raises(SystemExit):
        convert(tmp_path, tmp_path / "ann")   # un dossier au lieu d'un .xlsx
