"""Point d'entrée de l'API.

    uvicorn main:app --reload
    python main.py

La documentation interactive est sur /docs.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routers import documents, health, parcels, review

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
