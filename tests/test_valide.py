import pytest

from model.schema import Area, BoundaryPoint, Extracted, Extraction, Severity
from model.validate import shoelace_area, side_length, validate, validated

BASE_X, BASE_Y = 300_000.0, 1_600_000.0


def ex(v):
    return Extracted[float](value=v, confidence=0.95)


def pt(i, x, y):
    return BoundaryPoint(label=f"B{i + 1}", row_index=i,
                         x=ex(x) if x is not None else Extracted[float](),
                         y=ex(y) if y is not None else Extracted[float]())


def square(side=100.0, area=None, dx=0.0):
    pts = [pt(0, BASE_X, BASE_Y), pt(1, BASE_X + side + dx, BASE_Y),
           pt(2, BASE_X + side, BASE_Y + side), pt(3, BASE_X, BASE_Y + side)]
    printed = Extracted[Area](value=Area(ca=int(area if area is not None else side * side)), confidence=0.9)
    return Extraction(document_id="D", points=pts, printed_area=printed)


def codes(issues):
    return [i.code for i in issues]


def test_formule_de_gauss_et_precision():
    assert shoelace_area([(BASE_X, BASE_Y), (BASE_X + 10, BASE_Y), (BASE_X + 10, BASE_Y + 10),
                          (BASE_X, BASE_Y + 10)]) == pytest.approx(100.0)
    assert side_length((0, 0), (3, 4)) == 5.0


def test_parcelle_coherente_sans_anomalie():
    assert validate(square()) == []


def test_ecart_de_superficie():
    e = square(side=100, area=11_580)      # ~13,7 % d'écart
    assert "AREA_MISMATCH" in codes(validate(e))


def test_tolerance_de_superficie():
    assert validate(square(area=10_040)) == []                       # 0,4 % : accepté
    assert "AREA_MISMATCH" in codes(validate(square(area=10_100)))   # 1 % : refusé


def test_superficie_non_lue_est_signalee_sans_erreur():
    e = square()
    e.printed_area = Extracted[Area]()
    (i,) = validate(e)
    assert i.code == "AREA_UNAVAILABLE" and i.severity is Severity.INFO


def test_coordonnee_hors_plage_bloque_la_geometrie():
    e = square()
    e.points[1] = pt(1, 84_300_000.0, BASE_Y)
    c = codes(validate(e))
    assert "COORD_OUT_OF_RANGE" in c and "GEOMETRY_SKIPPED" in c and "AREA_MISMATCH" not in c


def test_valeur_aberrante_dans_la_plage():
    e = square()
    e.points[2] = pt(2, BASE_X + 50_000, BASE_Y + 100)   # plausible seule, absurde pour cette parcelle
    assert "COORD_OUTLIER" in codes(validate(e))


def test_point_incomplet():
    e = square()
    e.points[1] = pt(1, BASE_X + 100, None)
    assert "POINT_INCOMPLETE" in codes(validate(e))


def test_trop_peu_de_points():
    e = Extraction(document_id="D", points=[pt(0, BASE_X, BASE_Y), pt(1, BASE_X + 5, BASE_Y)])
    assert "TOO_FEW_POINTS" in codes(validate(e))


def test_polygone_croise():
    pts = [pt(0, BASE_X, BASE_Y), pt(1, BASE_X + 100, BASE_Y + 100),
           pt(2, BASE_X + 100, BASE_Y), pt(3, BASE_X, BASE_Y + 100)]   # papillon
    e = Extraction(document_id="D", points=pts, printed_area=Extracted[Area](value=Area(ca=5000), confidence=0.9))
    assert "POLYGON_INVALID" in codes(validate(e))


def test_point_de_fermeture_repete_est_ignore():
    e = square()
    e.points.append(pt(4, BASE_X, BASE_Y))
    assert validate(e) == []


def test_cotes_coherents():
    e = square()
    e.printed_sides = [ex(100.0)] * 4
    assert validate(e) == []


def test_sommet_suspect_localise_par_deux_cotes_adjacents():
    e = square(dx=10, area=10_500)         # B2 mal lu : côtés B1-B2 et B2-B3 faux
    e.printed_sides = [ex(100.0)] * 4
    issues = validate(e)
    suspects = [i for i in issues if i.code == "POINT_SUSPECT"]
    assert [s.point_labels for s in suspects] == [["B2"]]
    assert sum(1 for i in issues if i.code == "SIDE_MISMATCH") == 2


def test_un_seul_cote_faux_ne_designe_pas_de_suspect():
    e = square()
    e.printed_sides = [ex(100.0), ex(100.0), ex(100.0), ex(97.0)]
    c = codes(validate(e))
    assert c == ["SIDE_MISMATCH"]


def test_validated_ajoute_sans_toucher_loriginal():
    e = square(area=11_580)
    v = validated(e)
    assert e.issues == [] and "AREA_MISMATCH" in codes(v.issues) and v.needs_review()
