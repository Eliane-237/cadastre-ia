import json

import cv2
import numpy as np
import pytest

from model.__main__ import main as cli
from model.extract import run
from model.schema import Extraction, Source
from tests.helpers import COORDS, FakeReader, reading


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "D01.png"
    img = np.full((800, 600, 3), 250, np.uint8)
    cv2.putText(img, "PLAN DE BORNAGE", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.imwrite(str(path), img)
    return path


def test_chaine_complete_avec_lecteurs_factices(scan):
    p, q = FakeReader(reading(Source.PADDLE_VL)), FakeReader(reading(Source.QWEN_VL))
    result = run(scan, paddle=p, qwen=q)
    ex = result.extraction
    assert ex.document_id == "D01"
    assert [i.code for i in ex.issues] == []
    assert not ex.needs_review()
    prov = ex.provenance
    assert len(prov.input_sha256) == 64
    assert set(prov.models) == {"paddle_vl", "qwen_vl"}
    assert "illumination_unevenness" in prov.preprocessing["pages"][0]
    assert set(result.readings) == {"paddle_vl", "qwen_vl"}
    assert Extraction.model_validate_json(ex.model_dump_json()) == ex


def test_qwen_lit_limage_reorientee_par_paddle(scan):
    r = reading(Source.PADDLE_VL)
    r.image = np.zeros((10, 20, 3), np.uint8)  # page « tournée » par PaddleOCR
    q = FakeReader(reading(Source.QWEN_VL))
    run(scan, paddle=FakeReader(r), qwen=q)
    assert q.seen == [(10, 20, 3)]


def test_un_lecteur_en_panne_ne_bloque_pas_lautre(scan):
    ex = run(
        scan,
        paddle=FakeReader(RuntimeError("serveur injoignable")),
        qwen=FakeReader(reading(Source.QWEN_VL)),
    ).extraction
    assert "READER_FAILED" in [i.code for i in ex.issues]
    assert "paddle" in ex.provenance.errors
    assert ex.points[0].x.source is Source.QWEN_VL
    assert ex.needs_review()  # sans seconde lecture, tout est à relire


def test_les_deux_lecteurs_en_panne(scan):
    ex = run(
        scan, paddle=FakeReader(RuntimeError("a")), qwen=FakeReader(RuntimeError("b"))
    ).extraction
    assert "EXTRACTION_FAILED" in [i.code for i in ex.issues] and ex.has_error


def test_incoherence_geometrique_signalee_meme_si_les_lecteurs_saccordent(scan):
    wrong = [list(c) for c in COORDS]
    wrong[2][1] = "1564081.22"  # les deux lisent la même erreur du document
    wrong = [tuple(c) for c in wrong]
    ex = run(
        scan,
        paddle=FakeReader(reading(Source.PADDLE_VL, wrong)),
        qwen=FakeReader(reading(Source.QWEN_VL, wrong)),
    ).extraction
    codes = [i.code for i in ex.issues]
    assert "AREA_MISMATCH" in codes and "SIDE_MISMATCH" in codes
    assert ex.needs_review()


def test_pdf_multipage(tmp_path):
    import pymupdf

    path = tmp_path / "titre.pdf"
    doc = pymupdf.open()
    for _ in range(2):
        doc.new_page(width=200, height=300)
    doc.save(path)
    page1 = reading(Source.PADDLE_VL)
    page1.points = []
    page2 = reading(Source.PADDLE_VL)
    page2.fields = {}

    class TwoPages(FakeReader):
        def read(self, image):
            self.seen.append(image.shape)
            return [page1, page2][len(self.seen) - 1]

    ex = run(path, paddle=TwoPages(page1), qwen=FakeReader(reading(Source.QWEN_VL))).extraction
    assert len(ex.points) == 5 and ex.title_number.source is Source.CONSENSUS


def test_cli(scan, tmp_path, monkeypatch, capsys):
    import model.__main__ as m

    def fake_run(path, doc_id, **kw):
        return run(
            path,
            doc_id,
            paddle=FakeReader(reading(Source.PADDLE_VL)),
            qwen=FakeReader(reading(Source.QWEN_VL)),
        )

    monkeypatch.setattr(m, "run", fake_run)
    out, rd = tmp_path / "e.json", tmp_path / "r.json"
    assert cli([str(scan), "--out", str(out), "--readings", str(rd)]) == 0
    assert json.loads(out.read_text())["points"][0]["x"]["value"] == 381210.57
    assert set(json.loads(rd.read_text())) == {"paddle_vl", "qwen_vl"}
    assert "aucune revue requise" in capsys.readouterr().err
