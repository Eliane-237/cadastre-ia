"""Consigne et schéma imposés à Qwen3-VL.

Le schéma est appliqué au décodage (grammaire de llama-server) : le modèle ne peut produire
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

    # Identification du titre — une case du formulaire par champ.
    title_number: str | None
    mother_title: str | None
    lot: str | None
    subdivision: str | None
    requisition: str | None
    land_registry: str | None
    nicad: str | None

    # Localisation administrative — un niveau de la hiérarchie par champ.
    region: str | None
    department: str | None
    arrondissement: str | None
    commune: str | None

    owner: str | None
    coordinate_system: str | None
    scale: str | None
    certified_on: str | None
    disclaimer: str | None

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
- Chaque champ correspond à une case imprimée du formulaire. Recopie ce qui est écrit dans
  cette case précise, et rien d'autre : title_number est la case « TITRE N° », mother_title
  la case « Titre Mère n° », lot la case « Lot n° », region / department / arrondissement /
  commune les quatre lignes d'adresse administrative, owner la case « Requérant ».
- Si une case existe mais est vide, mets null. Ne va jamais chercher la valeur dans une
  autre case ni ailleurs sur le document. nicad est la seule case marquee « NICAD » : un
  autre nombre imprime sur la feuille n'est pas un NICAD.
- title_number et mother_title sont DEUX cases distinctes, et un titre n'est jamais son
  propre titre mere : ne mets jamais la meme valeur dans les deux. Si un seul numero est
  imprime, il va dans la case ou il est ecrit, et l'autre champ vaut null.
- Recopie la VALEUR, pas l'etiquette : si la ligne porte « Region de Thies », region vaut
  « Thies ». Si elle porte « Commune de Thies Nord », commune vaut « Thies Nord ».
- Si une valeur est rayee, biffee au marqueur, recouverte par un cachet ou masquee d'une
  quelconque maniere, mets null. Une valeur cachee n'est pas une valeur devinable.
- coordinate_system : uniquement ce qui est ecrit a cote de la mention « Systeme de
  coordonnees » ou « Systeme », recopie tel quel. Si le document ne porte aucune mention de
  ce genre, mets null. Ne le deduis jamais de l'allure des coordonnees, et ne reprends
  jamais un nom de systeme lu ailleurs que sur ce document.
- area : la superficie imprimée (souvent « 00 ha 06 a 49 ca »). Remplis text avec le texte exact, \
et ha, a, ca avec les nombres lus.
- points : toutes les lignes du tableau de coordonnées, dans l'ordre du tableau. X (Est) a \
6 chiffres avant la virgule, Y (Nord) en a 7 et commence par 1. \
Recopie chaque chiffre exactement, avec toutes les décimales imprimées, y compris les zéros \
finaux. Utilise le point comme séparateur décimal, sans espace.
- sides : les distances entre bornes si elles sont imprimées (tableau ou plan), avec les libellés \
des deux bornes.
- Si une valeur est illisible ou absente, mets null. N'invente rien. Ne répète pas la première \
borne à la fin si elle n'est pas répétée dans le tableau."""


def _inline_refs(node, defs: dict):
    if isinstance(node, dict):
        if "$ref" in node:
            return _inline_refs(defs[node["$ref"].split("/")[-1]], defs)
        return {k: _inline_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def json_schema() -> dict:
    """Schéma JSON sans références ($ref) : accepté par la grammaire de llama.cpp."""
    schema = QwenOutput.model_json_schema()
    return _inline_refs(schema, schema.get("$defs", {}))
