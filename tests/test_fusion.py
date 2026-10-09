import pytest

from model.config import Settings
from model.fusion import fuse
from model.readers.base import FieldReading
from model.schema import Source
from model.validate import validate
from tests.helpers import AREA_M2, COORDS, SIDES, reading

CFG = Settings()


def codes(issues):
    return [i.code for i in issues]


def with_coord(coords, i, axis, value):
    out = [list(c) for c in coords]
    out[i][0 if axis == "x" else 1] = value
    return [tuple(c) for c in out]


def test_accord_des_deux_lecteurs():
    ex, rep = fuse("D", reading(Source.PADDLE_VL), reading(Source.QWEN_VL), CFG)
    assert rep.issues == []
    p = ex.points[0]
    assert p.x.value == 381210.57 and p.x.source is Source.CONSENSUS
    assert p.x.confidence == CFG.conf_agree
    assert [r.source for r in p.x.readings] == [Source.PADDLE_VL, Source.QWEN_VL]
    assert ex.title_number.source is Source.CONSENSUS and ex.title_number.value == "TF 1738/DK"
    assert (
        ex.owner.source is Source.QWEN_VL and ex.owner.confidence == 0.9
    )  # plafond lecture unique
    assert ex.printed_area.value.m2 == AREA_M2
    assert [s.value for s in ex.printed_sides] == [float(s) for s in SIDES]
    assert validate(ex, CFG) == []
    assert not ex.needs_review(CFG.review_threshold)


def test_desaccord_tranche_par_la_geometrie():
    # Qwen lit un 8 au lieu d'un 3 : la seule combinaison cohérente est celle de PaddleOCR-VL.
    wrong = with_coord(COORDS, 2, "y", "1564081.22")
    ex, rep = fuse("D", reading(Source.PADDLE_VL), reading(Source.QWEN_VL, wrong), CFG)
    y = ex.points[2].y
    assert (
        y.value == 1564051.22 and y.source is Source.GEOMETRY and y.confidence == CFG.conf_geometry
    )
    assert codes(rep.issues) == ["READING_CONFLICT_RESOLVED"]
    assert rep.conflicts_resolved == 1


def test_la_geometrie_peut_donner_raison_a_qwen():
    wrong = with_coord(COORDS, 1, "x", "381255.56")
    ex, _ = fuse("D", reading(Source.PADDLE_VL, wrong), reading(Source.QWEN_VL), CFG)
    assert ex.points[1].x.value == 381225.56 and ex.points[1].x.raw_text == "381225.56"


def test_desaccord_sans_preuve_geometrique_part_en_revue():
    no_area = {"title_number": FieldReading("TF 1738/DK")}
    wrong = with_coord(COORDS, 2, "y", "1564081.22")
    ex, rep = fuse(
        "D",
        reading(Source.PADDLE_VL, sides=False, fields=no_area),
        reading(Source.QWEN_VL, wrong, sides=False, fields=no_area),
        CFG,
    )
    y = ex.points[2].y
    assert y.value == 1564051.22 and y.confidence == CFG.conf_conflict  # valeur Paddle, à relire
    (issue,) = rep.issues
    assert issue.code == "READING_CONFLICT" and issue.point_labels == ["B3"]
    assert ex.needs_review(CFG.review_threshold)
    assert "points[B3].y" in ex.low_confidence_fields(CFG.review_threshold)


def test_desaccord_sur_les_decimales_que_la_surface_ne_voit_pas():
    # 1 cm d'écart : la surface reste dans la tolérance pour les deux lectures, mais les côtés
    # imprimés (tolérance 5 cm) ne suffisent pas non plus : pas de choix unique → revue.
    wrong = with_coord(COORDS, 0, "x", "381210.58")
    ex, rep = fuse("D", reading(Source.PADDLE_VL), reading(Source.QWEN_VL, wrong), CFG)
    assert codes(rep.issues) == ["READING_CONFLICT"]
    assert ex.points[0].x.confidence == CFG.conf_conflict


def test_coordonnee_lue_une_seule_fois_toujours_relue_meme_si_qwen_est_sur():
    q = reading(Source.QWEN_VL)
    q.points[0].x_confidence = 0.999
    ex, _ = fuse("D", None, q, CFG)
    assert ex.points[0].x.confidence == CFG.conf_single_coordinate_cap < CFG.review_threshold
    assert ex.owner.confidence == CFG.conf_single_cap  # champ texte : plafond plus haut


def test_un_seul_lecteur():
    ex, rep = fuse("D", None, reading(Source.QWEN_VL), CFG)
    assert ex.points[0].x.source is Source.QWEN_VL
    assert ex.points[0].x.confidence == CFG.conf_single
    assert ex.needs_review(CFG.review_threshold)


def test_alignement_par_libelle_malgre_une_ligne_oubliee():
    q = reading(Source.QWEN_VL)
    del q.points[1]
    ex, rep = fuse("D", reading(Source.PADDLE_VL), q, CFG)
    assert "POINT_COUNT_MISMATCH" in codes(rep.issues)
    assert [p.label for p in ex.points] == ["B1", "B2", "B3", "B4", "B5"]
    assert ex.points[1].x.source is Source.PADDLE_VL  # lu par un seul
    assert ex.points[2].x.source is Source.CONSENSUS  # bien réaligné


def test_champ_divergent():
    q = reading(Source.QWEN_VL)
    q.fields["title_number"] = FieldReading("TF 1788/DK")
    ex, rep = fuse("D", reading(Source.PADDLE_VL), q, CFG)
    assert "FIELD_CONFLICT" in codes(rep.issues)
    assert ex.title_number.value == "TF 1788/DK" and ex.title_number.confidence == CFG.conf_conflict


def test_valeur_illisible_ignoree_mais_tracee():
    q = reading(Source.QWEN_VL, with_coord(COORDS, 0, "x", "illisible"))
    ex, _ = fuse("D", reading(Source.PADDLE_VL), q, CFG)
    x = ex.points[0].x
    assert x.source is Source.PADDLE_VL and x.value == 381210.57
    assert [r.raw_text for r in x.readings] == ["381210.57", "illisible"]


@pytest.mark.parametrize("dist", ["17.00", "17.01"])
def test_cotes_fusionnes(dist):
    q = reading(Source.QWEN_VL)
    q.sides[0].length = dist
    ex, _ = fuse("D", reading(Source.PADDLE_VL), q, CFG)
    s0 = ex.printed_sides[0]
    assert s0.source is (Source.CONSENSUS if dist == SIDES[0] else Source.PADDLE_VL)
