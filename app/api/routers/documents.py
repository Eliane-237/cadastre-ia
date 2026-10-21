"""Modules 1 à 3 — un document entre, une lecture jugée en sort.

Rien n'est enregistré ici. Lire et enregistrer sont deux gestes distincts : un agent
regarde d'abord ce qui a été lu, le corrige si besoin, et décide ensuite. Confondre les
deux, c'est laisser entrer en base tout ce qui passe par le lecteur.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.deps import settings
from app.api.schemas import JugementOut, LectureOut, ParcelleOut
from app.geospatial.parcel import GeometryRefusee, parcel
from app.validation.verdict import juger

router = APIRouter(prefix="/documents", tags=["document"])

FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".pdf"}


@router.post("", response_model=LectureOut, summary="Lire un titre foncier")
async def lire(fichier: UploadFile = File(...), document_id: str | None = None) -> LectureOut:
    """Photo, scan ou PDF → extraction, verdict, et polygone quand il tient.

    Synchrone : les deux lecteurs tournent pendant la requête, ce qui prend des secondes
    et non des millisecondes. À mettre en file d'attente le jour où plusieurs agents
    déposeront en même temps.
    """
    suffixe = Path(fichier.filename or "").suffix.lower()
    if suffixe not in FORMATS:
        raise HTTPException(415, f"format {suffixe or 'inconnu'} non pris en charge "
                                 f"(attendu : {', '.join(sorted(FORMATS))})")

    from app.extraction.pipeline import run  # importé ici : charge les lecteurs

    with tempfile.TemporaryDirectory() as dossier:
        chemin = Path(dossier) / (fichier.filename or f"document{suffixe}")
        chemin.write_bytes(await fichier.read())
        try:
            resultat = run(chemin, document_id, cfg=settings())
        except Exception as exc:                        # noqa: BLE001
            raise HTTPException(502, f"lecture impossible : {type(exc).__name__}: {exc}") from exc

    extraction = resultat.extraction
    jugement = juger(extraction)
    try:
        p = parcel(extraction)
        return LectureOut(extraction=extraction, jugement=JugementOut.de(jugement),
                          parcelle=ParcelleOut.de(p))
    except GeometryRefusee as exc:
        return LectureOut(extraction=extraction, jugement=JugementOut.de(jugement),
                          geometrie_refusee=str(exc))
