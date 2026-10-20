from pathlib import Path

import pytest
from openpyxl import Workbook
from pydantic import ValidationError

from eval.xlsx_to_annotations import convert
from model.annotation import (
    AnnotatedPoint,
    Annotation,
    NoValue,
    load_all,
    save_annotation,
)


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

def test_annotation_couvre_les_champs_du_schema():
    from model.schema import TEXT_FIELDS
    manquants = [f for f in TEXT_FIELDS if f not in Annotation.model_fields]
    assert not manquants, f"champs absents de l'annotation : {manquants}"

def _une(**kw) -> Annotation:
    return Annotation(document_id="D", image="a.png", **kw)


def test_champ_declare_sans_valeur_nest_plus_un_oubli():
    a = _une(no_value={"region": NoValue.ABSENT})
    assert "region" not in a.not_annotated
    assert "commune" in a.not_annotated        # ni rempli ni declare : un oubli


def test_champ_rempli_nest_pas_un_oubli():
    assert "commune" not in _une(commune="Kaolack").not_annotated


def test_champ_inconnu_refuse():
    with pytest.raises(ValidationError):
        _une(no_value={"longueurs_cotes": NoValue.ABSENT})


def test_raison_inconnue_refusee():
    with pytest.raises(ValidationError):
        _une(no_value={"region": "pas envie"})


def test_contradiction_refusee():
    # declarer sans valeur ce qu'on a soi-meme lu sur le document
    with pytest.raises(ValidationError):
        _une(commune="Kaolack", no_value={"commune": NoValue.ABSENT})


def test_superficie_declarable():
    assert "printed_area" not in _une(no_value={"printed_area": NoValue.ABSENT}).not_annotated


def test_seul_absent_est_mesurable():
    a = _une(
        commune="Kaolack",
        no_value={
            "region": NoValue.ABSENT,
            "title_number": NoValue.ILLISIBLE,
            "nicad": NoValue.HORS_CADRAGE,
            "arrondissement": NoValue.FUSIONNE,
        },
    )
    mesurables = a.measurable_fields
    assert "commune" in mesurables            # on sait quoi lire
    assert "region" in mesurables             # on sait qu'il ne faut rien lire
    for hors in ("title_number", "nicad", "arrondissement"):
        assert hors not in mesurables         # la valeur existe, notre vue ne l'atteint pas
    assert "owner" not in mesurables          # pas encore annote


def test_no_value_survit_a_laller_retour(tmp_path: Path):
    save_annotation(_une(no_value={"region": NoValue.ABSENT,
                                   "scale": NoValue.HORS_CADRAGE}), tmp_path)
    (b,) = load_all(tmp_path)
    assert b.no_value == {"region": NoValue.ABSENT, "scale": NoValue.HORS_CADRAGE}
