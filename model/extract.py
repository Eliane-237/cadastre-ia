"""Point d'entrée du modèle : document (image ou PDF) → `Extraction` validée.

    image ─► prétraitement (perspective, éclairage)
          ─► PaddleOCR-VL-1.6  (orientation, blocs, tableau de coordonnées en HTML)
          ─► Qwen3-VL-8B       (JSON imposé : tous les champs, tableau compris)
          ─► fusion            (consensus / arbitrage géométrique / revue humaine)
          ─► validate()        (Gauss, plage UTM, côtés, polygone)

Si un lecteur échoue, l'autre suffit à produire un résultat, mais sans recoupement : les
confiances restent sous le seuil et le document part en revue.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from model.config import SETTINGS, Settings
from model.fusion import fuse
from model.imaging import file_sha256, load_pages, preprocess
from model.readers.base import DocumentReading, Reader, merge_pages
from model.schema import Extraction, Issue, Severity
from model.validate import validate


@dataclass
class ExtractionRun:
    extraction: Extraction
    readings: dict[str, DocumentReading] = field(default_factory=dict)


def default_readers(cfg: Settings = SETTINGS) -> tuple[Reader, Reader]:
    from model.readers.paddle_vl import PaddleVLReader
    from model.readers.qwen_vl import QwenVLReader

    return PaddleVLReader(cfg.paddle), QwenVLReader(cfg.qwen)


def run(
    image_path: Path,
    document_id: str | None = None,
    *,
    paddle: Reader | None = None,
    qwen: Reader | None = None,
    cfg: Settings = SETTINGS,
    use_paddle: bool = True,
    use_qwen: bool = True,
) -> ExtractionRun:
    image_path = Path(image_path)
    document_id = document_id or image_path.stem
    if (use_paddle and paddle is None) or (use_qwen and qwen is None):
        dp, dq = default_readers(cfg)
        paddle = paddle or dp
        qwen = qwen or dq

    timings: dict[str, float] = {}
    errors: dict[str, str] = {}
    pre_log = []
    paddle_pages: list[DocumentReading] = []
    qwen_pages: list[DocumentReading] = []

    t0 = time.perf_counter()
    pages = load_pages(image_path)
    for index, raw in enumerate(pages):
        image, steps = preprocess(raw)
        pre_log.append({"page": index, **steps})
        if use_paddle and "paddle" not in errors:
            t = time.perf_counter()
            try:
                pr = paddle.read(image)  # type: ignore[union-attr]
                paddle_pages.append(pr)
                if pr.image is not None:  # PaddleOCR a réorienté la page : Qwen lit la même
                    image = pr.image
                pre_log[-1]["orientation_deg"] = pr.orientation_deg
            except Exception as exc:  # noqa: BLE001 - un lecteur en panne ne bloque pas l'autre
                errors["paddle"] = f"{type(exc).__name__}: {exc}"
            timings["paddle_vl"] = timings.get("paddle_vl", 0.0) + time.perf_counter() - t
        if use_qwen and "qwen" not in errors:
            t = time.perf_counter()
            try:
                qwen_pages.append(qwen.read(image))  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                errors["qwen"] = f"{type(exc).__name__}: {exc}"
            timings["qwen_vl"] = timings.get("qwen_vl", 0.0) + time.perf_counter() - t

    pr = merge_pages(paddle_pages) if paddle_pages and "paddle" not in errors else None
    qr = merge_pages(qwen_pages) if qwen_pages and "qwen" not in errors else None
    extraction, report = fuse(document_id, pr, qr, cfg)
    issues = list(report.issues)
    for name, err in errors.items():
        issues.append(
            Issue(
                code="READER_FAILED",
                severity=Severity.WARNING,
                message=f"Lecteur {name} en échec ({err}) : pas de seconde lecture.",
            )
        )
    if pr is None and qr is None:
        issues.append(
            Issue(
                code="EXTRACTION_FAILED",
                severity=Severity.ERROR,
                message="Aucun lecteur n'a pu lire le document.",
            )
        )
    else:
        issues.extend(validate(extraction, cfg))
    extraction.issues = issues

    timings["total"] = time.perf_counter() - t0
    prov = extraction.provenance
    prov.input_sha256 = file_sha256(image_path)
    prov.models = {r.source.value: r.model for r in (pr, qr) if r is not None}
    prov.preprocessing = {"pages": pre_log}
    prov.timings_s = {k: round(v, 3) for k, v in timings.items()}
    prov.errors = errors
    readings = {r.source.value: r for r in (pr, qr) if r is not None}
    return ExtractionRun(extraction=extraction, readings=readings)


def extract(image_path: Path, document_id: str | None = None, **kwargs) -> Extraction:
    return run(image_path, document_id, **kwargs).extraction
