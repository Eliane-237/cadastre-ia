"""Point d'entrée de l'API.

    python main.py
    uvicorn main:app --reload --port 8800

L'interface des agents est sur http://127.0.0.1:8800/ et la documentation
interactive sur http://127.0.0.1:8800/docs

Le port 8800 n'est pas un choix de confort : 8000 et 8080 sont tenus par les deux
serveurs de modèles (Qwen3-VL et PaddleOCR-VL). Y poser l'API ferait répondre
llama-server à sa place.

"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routers import documents, health, parcels, review

WEB = Path(__file__).parent / "web"

HOTE = os.getenv("CADASTRE_API_HOST", "127.0.0.1")
PORT = int(os.getenv("CADASTRE_API_PORT", "8800"))

app = FastAPI(
    title="cadastre-ia",
    version="0.1.0",
    summary="Numérisation et contrôle des titres fonciers sénégalais.",
    description=(
        "Un document entre par `/documents`, il en sort une extraction et un verdict. "
        "Rien n'est enregistré à ce moment-là : lire et enregistrer sont deux gestes "
        "distincts. L'agent confronte la parcelle à la base par `/parcelles/confrontation`, "
        "puis l'enregistre par `/parcelles` — et seul un verdict « exploitable » passe.\n\n"
        "Le système **ne déclare jamais une fraude**. Il constate un recouvrement "
        "géométrique et le qualifie ; dire s'il s'agit d'une fraude, d'une erreur de levé "
        "ou d'un morcellement relève de l'humain et du droit."
    ),
)

for routeur in (health.router, documents.router, parcels.router, review.router):
    app.include_router(routeur)


# L'interface est servie par cette API, et non ouverte comme un fichier local :
# une page sur file:// n'a pas le droit d'appeler http://127.0.0.1:8800. Même
# origine, donc aucun CORS à configurer.
@app.get("/", include_in_schema=False)
def accueil() -> RedirectResponse:
    return RedirectResponse("/web/")


app.mount("/web", StaticFiles(directory=WEB, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    print(f"interface     : http://{HOTE}:{PORT}/")
    print(f"documentation : http://{HOTE}:{PORT}/docs")
    uvicorn.run("main:app", host=HOTE, port=PORT, reload=True)
