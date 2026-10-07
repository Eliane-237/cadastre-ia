import pytest
from pydantic import ValidationError

from model.schema import (
    Area,
    BBox,
    BoundaryPoint,
    Extracted,
    Extraction,
    Issue,
    Severity,
    Source,
)


def ex(v, c=0.95):
    return Extracted[float](value=v, confidence=c, source=Source.OCR)


def test_area_normalisee_en_m2():
    assert Area(ha=3, a=12, ca=49).m2 == 31_249
    assert Area(a=6, ca=49).m2 == 649


def test_bbox_incoherente_refusee():
    with pytest.raises(ValidationError):
        BBox(x0=10, y0=10, x1=5, y1=20)


def test_confiance_bornee():
    with pytest.raises(ValidationError):
        Extracted[float](value=1.0, confidence=1.2)


def test_champ_absent_nest_pas_fiable():
    assert not Extracted[str]().is_reliable()


def test_revue_si_champ_manquant_ou_peu_fiable():
    e = Extraction(document_id="D01")
    assert e.needs_review()
    assert "title_number" in e.low_confidence_fields()


def test_point_peu_fiable_est_signale():
    p = BoundaryPoint(label="B1", x=ex(234773.60, 0.95), y=ex(1627666.10, 0.40), row_index=0)
    e = Extraction(document_id="D05", points=[p])
    assert "points[B1].y" in e.low_confidence_fields()
    assert "points[B1].x" not in e.low_confidence_fields()


def test_erreur_force_la_revue_meme_si_tout_est_fiable():
    full = Extracted[str](value="x", confidence=0.99)
    e = Extraction(
        document_id="D02", title_number=full, nicad=full, owner=full, locality=full,
        printed_area=Extracted[Area](value=Area(ca=2100), confidence=0.99),
    )
    assert not e.needs_review()
    e.issues.append(Issue(code="AREA_MISMATCH", severity=Severity.ERROR, message="écart 13,7 %"))
    assert e.needs_review()


def test_aller_retour_json():
    e = Extraction(document_id="D01", points=[
        BoundaryPoint(label="B1", x=ex(381210.57), y=ex(1564042.96), row_index=0)])
    assert Extraction.model_validate_json(e.model_dump_json()) == e
