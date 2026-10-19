"""Exécution sous Windows avec llama.cpp : réglages, passes séparées, schéma sans $ref."""

import json
import sys
import types

import numpy as np

from eval import evaluate
from model.config import PaddleSettings, from_env
from model.extract import run
from model.paddle_reader import PaddleVLReader
from model.preprocess import apply_orientation
from model.qwen_prompt import json_schema
from model.schema import Source
from tests.fake_readers import FakeReader, reading
from tests.test_extract import scan  # noqa: F401 - fixture


def test_reglages_par_lenvironnement(monkeypatch):
    monkeypatch.setenv("CADASTRE_PADDLE_SERVER_KIND", "llama-cpp-server")
    monkeypatch.setenv("CADASTRE_PADDLE_DEVICE", "cpu")
    cfg = from_env()
    assert cfg.paddle.server_kind == "llama-cpp-server" and cfg.paddle.device == "cpu"


def test_paddle_branche_sur_llama_cpp(monkeypatch):
    seen = {}
    fake = types.ModuleType("paddleocr")
    fake.PaddleOCRVL = lambda **kw: seen.update(kw) or object()
    monkeypatch.setitem(sys.modules, "paddleocr", fake)
    assert PaddleVLReader(PaddleSettings(server_kind="llama-cpp-server", device="cpu")).pipeline
    assert seen["vl_rec_backend"] == "llama-cpp-server"
    assert seen["vl_rec_server_url"] == "http://127.0.0.1:8080/v1" and seen["device"] == "cpu"


def test_schema_sans_reference():
    text = json.dumps(json_schema())
    assert "$ref" not in text and "$defs" not in text
    assert json_schema()["properties"]["points"]["items"]["properties"]["x"]


def test_meme_rotation_que_paddleocr():
    img = np.arange(6, dtype=np.uint8).reshape(2, 3, 1)
    assert apply_orientation(img, 90).shape == (3, 2, 1)
    assert apply_orientation(img, 90)[0, 0, 0] == img[0, 2, 0]  # sens anti-horaire
    assert np.array_equal(apply_orientation(img, 180), img[::-1, ::-1])
    assert apply_orientation(img, None) is img


def test_seconde_passe_qwen_reprend_lorientation(scan):  # noqa: F811
    q = FakeReader(reading(Source.QWEN_VL))
    run(scan, paddle=None, qwen=q, use_paddle=False, orientation_deg=90)
    assert q.seen == [(600, 800, 3)]  # page 800×600 tournée de 90°


def test_deux_passes_completent_les_lectures(tmp_path, monkeypatch):
    from model.annotation import AnnotatedPoint, Annotation, save_annotation

    raw, ann, out = tmp_path / "raw", tmp_path / "ann", tmp_path / "rep"
    raw.mkdir()
    (raw / "D01.png").write_bytes(b"")
    save_annotation(
        Annotation(
            document_id="D01",
            image="D01.png",
            points=[AnnotatedPoint(label="B1", x="381210.57", y="1564042.96")],
        ),
        ann,
    )
    p = reading(Source.PADDLE_VL)
    p.orientation_deg = 180
    calls = []

    def fake_run(img, doc_id, *, cfg, use_paddle, use_qwen, orientation_deg):
        calls.append((use_paddle, use_qwen, orientation_deg))
        readings = {}
        if use_paddle:
            readings["paddle_vl"] = p
        if use_qwen:
            readings["qwen_vl"] = reading(Source.QWEN_VL)
        return types.SimpleNamespace(readings=readings)

    monkeypatch.setattr(evaluate, "run", fake_run)
    evaluate.main(raw, ann, out, replay=False, only="paddle")
    evaluate.main(raw, ann, out, replay=False, only="qwen")
    assert calls == [(True, False, None), (False, True, 180)]
    cached = json.loads((out / "readings" / "D01.json").read_text())
    assert set(cached) == {"paddle_vl", "qwen_vl"}
    report = json.loads((out / "evaluation.json").read_text())
    assert report["summary"]["fusion"]["values"] == 2


def test_verification_des_serveurs(capsys):
    import httpx

    from model import check_servers
    from model.config import Settings

    def handler(request):
        return httpx.Response(200 if request.url.port == 8080 else 503)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert check_servers.main(Settings(), client) == 1
    out = capsys.readouterr().out
    assert "✓ PaddleOCR-VL prêt sur http://127.0.0.1:8080" in out
    assert "Qwen3-VL démarre encore" in out
