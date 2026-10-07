import pytest

from eval.metrics import (
    FieldResult, Outcome, calibration_table, compare, levenshtein, summarize, threshold_for,
)
from model.annotation import AnnotatedPoint, Annotation
from model.schema import Area, BoundaryPoint, Extracted, Extraction


def pt(i, x, y, cx=0.9, cy=0.9):
    return BoundaryPoint(label=f"B{i+1}", row_index=i,
                         x=Extracted[float](value=x, confidence=cx),
                         y=Extracted[float](value=y, confidence=cy))


TRUTH = Annotation(
    document_id="D", image="d.png", decimals=2, title_number="TF 12/DK", printed_area=Area(a=6, ca=49),
    points=[AnnotatedPoint(label="B1", x="381210.57", y="1564042.96"),
            AnnotatedPoint(label="B2", x="381225.56", y="1564034.94")],
)


def outcomes(rep):
    return {r.field: r.outcome for r in rep.results}


def test_levenshtein():
    assert levenshtein("1234", "1234") == 0
    assert levenshtein("1234", "1284") == 1
    assert levenshtein("1234", "12") == 2


def test_extraction_parfaite():
    e = Extraction(document_id="D", points=[pt(0, 381210.57, 1564042.96), pt(1, 381225.56, 1564034.94)],
                   title_number=Extracted[str](value="tf 12/dk ", confidence=0.9),
                   printed_area=Extracted[Area](value=Area(a=6, ca=49), confidence=0.9))
    s = summarize([compare(e, TRUTH)])
    assert s["exact_rate"] == 1.0 and s["values"] == 6


def test_quatre_issues_distinguees():
    e = Extraction(document_id="D", points=[
        pt(0, 381210.57, 1564042.96),
        pt(1, 381225.58, 84034.94),         # x : un chiffre faux (near) ; y : très faux (wrong)
    ])
    o = outcomes(compare(e, TRUTH))
    assert o["points[0].x"] is Outcome.EXACT
    assert o["points[1].x"] is Outcome.NEAR
    assert o["points[1].y"] is Outcome.WRONG
    assert o["title_number"] is Outcome.MISSING      # attendu mais non lu


def test_point_manquant_et_point_en_trop():
    e = Extraction(document_id="D", points=[pt(0, 381210.57, 1564042.96), pt(2, 1.0, 2.0)])
    rep = compare(e, TRUTH)
    assert outcomes(rep)["points[1].x"] is Outcome.MISSING
    assert rep.spurious_points == 1


def test_zero_final_preserve():
    t = Annotation(document_id="D", image="d.png", decimals=3,
                   points=[AnnotatedPoint(label="B1", x="235906.620", y="1624690.747")])
    e = Extraction(document_id="D", points=[pt(0, 235906.62, 1624690.747)])
    assert summarize([compare(e, t)])["exact_rate"] == 1.0


def _fr(conf, ok):
    return FieldResult("D", "f", "a", "a" if ok else "b", conf, Outcome.EXACT if ok else Outcome.WRONG)


def test_calibration_et_seuil():
    res = [_fr(0.6, False), _fr(0.65, True), _fr(0.9, True), _fr(0.92, True), _fr(0.97, True)]
    rows = calibration_table(res)
    assert rows[1]["n"] == 2 and rows[1]["errors"] == 1          # tranche 0,5–0,7
    t = threshold_for(res, max_error_rate=0.0)
    assert t["threshold"] == 0.65 and t["accepted"] == 4
    assert t["review_share"] == pytest.approx(0.2)


def test_seuil_introuvable():
    assert threshold_for([_fr(0.9, False)], 0.0)["threshold"] is None
