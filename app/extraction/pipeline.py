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

from app.core.config import SETTINGS, Settings
from app.core.schema import Extraction, Issue, Severity
from app.document.loader import file_sha256, load_pages
from app.document.preprocess import apply_orientation, preprocess
from app.extraction.locators.area import best_region as aire_best_region
from app.extraction.locators.table import best_region, crop_region
from app.extraction.merge import fuse
from app.extraction.reading import DocumentReading, Reader, merge_pages
from app.validation.localites import localite_connue
from app.validation.rules import referentiel_de, validate


@dataclass
class ExtractionRun:
    extraction: Extraction
    readings: dict[str, DocumentReading] = field(default_factory=dict)


def default_readers(cfg: Settings = SETTINGS) -> tuple[Reader, Reader]:
    from app.extraction.readers.paddle import PaddleVLReader
    from app.extraction.readers.qwen import QwenVLReader

    return PaddleVLReader(cfg.paddle), QwenVLReader(cfg.qwen)


def _aire_lue(*listes: list[DocumentReading]) -> bool:
    """Un des lecteurs a-t-il deja vu la superficie sur la page entiere ?"""
    return any(r.field("area").raw for liste in listes for r in liste)


def _aire_seule(r: DocumentReading) -> DocumentReading:
    """D'une relecture de fragment, ne garder que la superficie.

    Le fragment ne montre qu'un coin de la feuille. Tout autre champ que le lecteur
    croirait y reconnaitre serait invente, et `merge_pages` pourrait le retenir pour un
    champ que la page entiere n'a pas donne. On jette aussi les points : le tableau de
    coordonnees n'est pas dans ce fragment.
    """
    aire = r.field("area")
    return DocumentReading(
        source=r.source,
        model=r.model,
        fields={"area": aire} if aire.raw else {},
        raw_output=r.raw_output,
        orientation_deg=r.orientation_deg,
    )


def run(
    image_path: Path,
    document_id: str | None = None,
    *,
    paddle: Reader | None = None,
    qwen: Reader | None = None,
    cfg: Settings = SETTINGS,
    use_paddle: bool = True,
    use_qwen: bool = True,
    orientation_deg: int | None = None,
) -> ExtractionRun:
    """`orientation_deg` : orientation trouvée par PaddleOCR-VL lors d'une passe précédente
    (lecture en deux passes sur un petit GPU) ; appliquée à l'image avant Qwen3-VL."""
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
        if not use_paddle and orientation_deg:
            image = apply_orientation(image, orientation_deg)
            pre_log[-1]["orientation_deg"] = orientation_deg
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

        # Relecture ciblee du tableau de coordonnees. La page entiere vient d'etre lue et
        # garde tous ses champs ; on relit en plus la seule region du tableau, ou les
        # chiffres occupent toute la resolution du modele au lieu de quelques pour cent de
        # la feuille. `merge_pages` retiendra ensuite le tableau le plus complet des deux.
        if cfg.table_rescan:
            region = best_region(paddle_pages[-1].raw_output if paddle_pages else None, image)
            pre_log[-1]["table_region"] = (
                {"bbox": [round(v, 1) for v in region.bbox], "source": region.source}
                if region is not None
                else None
            )
            if region is not None:
                fragment = crop_region(image, region.bbox, cfg.table_zoom)
                if fragment is not None:
                    for actif, lecteur, liste, cle in (
                        (use_paddle, paddle, paddle_pages, "paddle"),
                        (use_qwen, qwen, qwen_pages, "qwen"),
                    ):
                        if not actif or cle in errors or lecteur is None:
                            continue
                        t = time.perf_counter()
                        try:
                            # Une relecture ciblee qui echoue ne doit rien coûter : la
                            # lecture de page reste acquise.
                            liste.append(lecteur.read(fragment))
                        except Exception as exc:  # noqa: BLE001
                            pre_log[-1][f"table_rescan_{cle}"] = f"{type(exc).__name__}: {exc}"
                        timings[f"{cle}_table"] = (
                            timings.get(f"{cle}_table", 0.0) + time.perf_counter() - t
                        )

        # Relecture ciblee de la superficie. Elle juge toute la validation geometrique :
        # sans elle, l'aire de Gauss n'est confrontee a rien et le document ne peut plus se
        # contredire lui-meme. On ne paie cette relecture que si la page entiere n'a rien
        # donne — quand elle a lu la superficie, `merge_pages` la garderait de toute facon.
        if cfg.area_rescan and not _aire_lue(paddle_pages, qwen_pages):
            region = aire_best_region(paddle_pages[-1].raw_output if paddle_pages else None,
                                      image)
            pre_log[-1]["area_region"] = (
                {"bbox": [round(v, 1) for v in region.bbox], "source": region.source}
                if region is not None
                else None
            )
            if region is not None:
                fragment = crop_region(image, region.bbox, cfg.area_zoom)
                if fragment is not None:
                    for actif, lecteur, liste, cle in (
                        (use_paddle, paddle, paddle_pages, "paddle"),
                        (use_qwen, qwen, qwen_pages, "qwen"),
                    ):
                        if not actif or cle in errors or lecteur is None:
                            continue
                        t = time.perf_counter()
                        try:
                            liste.append(_aire_seule(lecteur.read(fragment)))
                        except Exception as exc:  # noqa: BLE001
                            pre_log[-1][f"area_rescan_{cle}"] = f"{type(exc).__name__}: {exc}"
                        timings[f"{cle}_area"] = (
                            timings.get(f"{cle}_area", 0.0) + time.perf_counter() - t
                        )

    pr = merge_pages(paddle_pages) if paddle_pages and "paddle" not in errors else None
    qr = merge_pages(qwen_pages) if qwen_pages and "qwen" not in errors else None
    # Le répertoire des localités est fourni ici, et non importé par la fusion : c'est
    # l'orchestrateur qui connaît la validation, pas l'extraction.
    extraction, report = fuse(document_id, pr, qr, cfg, connu=localite_connue)
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
    # Le fuseau ET le référentiel, lus séparément sur le système imprimé. Sans le datum
    # la parcelle reste « incomparable » et aucun chevauchement ne peut être cherché ;
    # sans le bon fuseau, elle se pose à des centaines de kilomètres de son terrain.
    extraction.crs, extraction.datum = referentiel_de(
        extraction.coordinate_system.value if extraction.coordinate_system.present else None
    )

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
