"""Consigne et schéma imposés à Qwen3-VL.

Le schéma est appliqué au décodage (sortie structurée de vLLM) : le modèle ne peut produire
que ce JSON. Les coordonnées sont des chaînes pour recopier exactement les chiffres imprimés.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QwenPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str | None = Field(description="Libellé du sommet tel qu'imprimé (B1, 12, S3…)")
    x: str | None = Field(description="Coordonnée X / Est, chiffres recopiés à l'identique")
    y: str | None = Field(description="Coordonnée Y / Nord, chiffres recopiés à l'identique")


class QwenSide(BaseModel):
    model_config = ConfigDict(extra="forbid")
    from_label: str | None
    to_label: str | None
    length: str | None = Field(description="Distance imprimée en mètres, recopiée à l'identique")


class QwenArea(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str | None = Field(description="Superficie recopiée telle qu'imprimée")
    ha: int | None
    a: int | None
    ca: int | None


class QwenOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_type: Literal["titre_foncier", "plan_bornage", "autre"] | None
    title_number: str | None
    nicad: str | None
    owner: str | None
    locality: str | None
    area: QwenArea | None
    points: list[QwenPoint]
    sides: list[QwenSide]


SYSTEM_PROMPT = (
    "Tu es un agent de saisie du cadastre du Sénégal. Tu recopies les informations d'un titre "
    "foncier ou d'un plan de bornage, sans jamais les corriger, les compléter ni les calculer."
)

USER_PROMPT = """Lis ce document et renvoie uniquement le JSON demandé.

Règles :
- title_number : numéro du titre foncier ou du titre mère tel qu'écrit (ex. « TF 1738/DK », \
« TM 1738/DK (ex 3016/DG) »).
- nicad : numéro NICAD (identifiant cadastral), chiffres seulement s'il est présent.
- owner : propriétaire ou requérant, tel qu'écrit. locality : commune, quartier ou lieu-dit.
- area : la superficie imprimée (souvent « 00 ha 06 a 49 ca »). Remplis text avec le texte exact, \
et ha, a, ca avec les nombres lus.
- points : toutes les lignes du tableau de coordonnées, dans l'ordre du tableau. Les coordonnées \
sont en UTM zone 28N : X (Est) a 6 chiffres avant la virgule, Y (Nord) en a 7 et commence par 1. \
Recopie chaque chiffre exactement, avec toutes les décimales imprimées, y compris les zéros \
finaux. Utilise le point comme séparateur décimal, sans espace.
- sides : les distances entre bornes si elles sont imprimées (tableau ou plan), avec les libellés \
des deux bornes.
- Si une valeur est illisible ou absente, mets null. N'invente rien. Ne répète pas la première \
borne à la fin si elle n'est pas répétée dans le tableau."""


def json_schema() -> dict:
    return QwenOutput.model_json_schema()
