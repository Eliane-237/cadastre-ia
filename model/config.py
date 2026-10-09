"""Paramètres de travail du modèle.

Aucune de ces valeurs n'est mesurée à ce stade : elles seront recalibrées sur les titres annotés
(voir eval/). On les garde ici, et nulle part ailleurs, pour pouvoir les changer en un point.

Les points d'accès aux modèles se surchargent par variables d'environnement (voir `from_env`),
pour passer d'un poste à un serveur GPU sans toucher au code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Literal

Backend = Literal["server", "local"]


@dataclass(frozen=True)
class PaddleSettings:
    """PaddleOCR-VL-1.6 : parsing de page (blocs + tableaux en structure)."""

    pipeline_version: str = "v1.6"
    # « server » : le VLM tourne derrière un serveur vLLM / genai_server (recommandé sur GPU) ;
    # « local » : tout dans le processus Python (plus simple, plus lent).
    backend: Backend = "server"
    server_url: str = "http://127.0.0.1:8080/v1"
    device: str | None = None  # ex. « gpu:0 » ; None = choix de PaddleOCR
    use_doc_orientation_classify: bool = True  # corrige 90/180/270°
    # Dépliage des pages courbées (UVDoc). Désactivé : notre redressement de perspective suffit
    # pour les photos à plat, et l'image brute donne de meilleurs résultats sur les scans.
    use_doc_unwarping: bool = False


@dataclass(frozen=True)
class QwenSettings:
    """Qwen3-VL-8B-Instruct : remplissage du JSON de l'`Extraction` (deuxième lecture)."""

    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    backend: Backend = "server"
    server_url: str = "http://127.0.0.1:8000/v1"  # API compatible OpenAI (vLLM)
    api_key: str = "EMPTY"
    timeout_s: float = 180.0
    max_new_tokens: int = 2048
    # Lecture déterministe : on veut recopier, pas créer.
    temperature: float = 0.0
    # Résolution maximale envoyée au modèle (côté long, en pixels). Les chiffres des tableaux
    # doivent rester lisibles : ne pas descendre sous ~1600.
    max_image_side: int = 2048


@dataclass(frozen=True)
class Settings:
    # Confiance minimale pour accepter une valeur sans relecture (à calibrer).
    review_threshold: float = 0.85
    # Tolérances de cohérence géométrique (à calibrer).
    side_tolerance_m: float = 0.05
    area_tolerance_pct: float = 0.5
    # Plages plausibles en UTM 28N pour le Sénégal (m). Larges à dessein : elles écartent les
    # lectures absurdes, pas les parcelles inhabituelles. À resserrer avec les données du cadastre.
    easting_range: tuple[float, float] = (100_000.0, 900_000.0)
    northing_range: tuple[float, float] = (1_300_000.0, 1_900_000.0)
    # Étendue maximale d'une parcelle (m) : au-delà, un sommet est probablement mal lu.
    max_parcel_extent_m: float = 10_000.0

    # Confiances attribuées par la fusion des deux lectures (à calibrer avec
    # eval.metrics.calibration_table : ce sont des rangs, pas encore des probabilités).
    conf_agree: float = 0.97  # PaddleOCR-VL et Qwen3-VL lisent la même chaîne
    conf_geometry: float = 0.90  # désaccord tranché par la géométrie (surface, côtés)
    conf_single: float = 0.60  # une seule lecture, sans confiance propre du lecteur
    conf_single_cap: float = 0.90  # plafond d'une lecture unique d'un champ texte
    # Plafond d'une coordonnée (ou d'une distance) lue par un seul lecteur : sous le seuil de
    # revue, car une seule lecture d'un chiffre n'est jamais une preuve suffisante.
    conf_single_coordinate_cap: float = 0.60
    conf_conflict: float = 0.30  # désaccord non tranché : revue humaine
    # Nombre maximal de coordonnées en désaccord pour lesquelles on teste toutes les
    # combinaisons (2^n) afin que la géométrie tranche.
    max_conflicts_for_geometry: int = 8

    paddle: PaddleSettings = field(default_factory=PaddleSettings)
    qwen: QwenSettings = field(default_factory=QwenSettings)


def from_env(base: Settings | None = None) -> Settings:
    """Surcharge les points d'accès par l'environnement (CADASTRE_QWEN_URL, etc.)."""
    s = base or Settings()
    env = os.environ
    paddle = replace(
        s.paddle,
        backend=env.get("CADASTRE_PADDLE_BACKEND", s.paddle.backend),  # type: ignore[arg-type]
        server_url=env.get("CADASTRE_PADDLE_URL", s.paddle.server_url),
        device=env.get("CADASTRE_PADDLE_DEVICE", s.paddle.device),
    )
    qwen = replace(
        s.qwen,
        backend=env.get("CADASTRE_QWEN_BACKEND", s.qwen.backend),  # type: ignore[arg-type]
        server_url=env.get("CADASTRE_QWEN_URL", s.qwen.server_url),
        model=env.get("CADASTRE_QWEN_MODEL", s.qwen.model),
        api_key=env.get("CADASTRE_QWEN_API_KEY", s.qwen.api_key),
    )
    return replace(s, paddle=paddle, qwen=qwen)


SETTINGS = Settings()
