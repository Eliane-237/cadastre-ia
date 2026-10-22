"""Modules 1 à 3 — un document entre, une lecture jugée en sort.

Rien n'est enregistré ici. Lire et enregistrer sont deux gestes distincts : un agent
regarde d'abord ce qui a été lu, le corrige si besoin, et décide ensuite. Confondre les
deux, c'est laisser entrer en base tout ce qui passe par le lecteur.
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.deps import settings
from app.api.schemas import JugementOut, LectureOut, ParcelleOut
from app.geospatial.export import centre_wgs84, to_geojson
from app.geospatial.parcel import GeometryRefusee, parcel
from app.geospatial.render import RenduIndisponible, apercu_png, libelles_imprimes
from app.validation.verdict import juger

router = APIRouter(prefix="/documents", tags=["document"])

FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".pdf"}


@router.post("", response_model=LectureOut, summary="Lire un titre foncier")
async def lire(fichier: UploadFile = File(...), document_id: str | None = None,
               apercu: bool = False) -> LectureOut:
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
    except GeometryRefusee as exc:
        # La lecture reste utile même sans polygone : les champs texte sont là, et le
        # message dit ce qui manque pour en construire un.
        return LectureOut(extraction=extraction, jugement=JugementOut.de(jugement),
                          geometrie_refusee=str(exc))

    dessin = None
    if apercu:
        try:
            dessin = base64.b64encode(
                apercu_png(p, libelles_imprimes(extraction))).decode()
        except RenduIndisponible:
            dessin = None        # un aperçu absent ne doit pas coûter la lecture
    return LectureOut(
        extraction=extraction, jugement=JugementOut.de(jugement), parcelle=ParcelleOut.de(p),
        geojson=to_geojson([p]), centre=centre_wgs84(p), apercu_png=dessin,
    )
