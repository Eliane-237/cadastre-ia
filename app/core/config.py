"""Paramètres de travail du modèle.

Aucune de ces valeurs n'est mesurée à ce stade : elles seront recalibrées sur les titres annotés
(voir eval/). On les garde ici, et nulle part ailleurs, pour pouvoir les changer en un point.

Les deux modèles tournent chacun derrière un serveur llama.cpp (`llama-server`, voir
scripts/). Leurs adresses se surchargent par variables d'environnement (voir `from_env`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class PaddleSettings:
    """PaddleOCR-VL-1.6 : parsing de page (blocs + tableaux en structure)."""

    pipeline_version: str = "v1.6"
    # Serveur du VLM : llama.cpp (scripts/start_paddle_server.ps1). PaddleOCR accepte aussi
    # « vllm-server » si on passe un jour sur un serveur Linux.
    server_kind: str = "llama-cpp-server"
    server_url: str = "http://127.0.0.1:8080/v1"
    # Où tourne la mise en page (PP-DocLayout) dans le processus : « gpu:0 », « cpu »…
    # « cpu » laisse toute la mémoire GPU aux deux VLM (petites cartes).
    device: str | None = None
    use_doc_orientation_classify: bool = True  # corrige 90/180/270°
    # Dépliage des pages courbées (UVDoc). Désactivé : notre redressement de perspective suffit
    # pour les photos à plat, et l'image brute donne de meilleurs résultats sur les scans.
    use_doc_unwarping: bool = False
    merge_layout_blocks: bool = False
    layout_merge_bboxes_mode: str = "small"


@dataclass(frozen=True)
class QwenSettings:
    """Qwen3-VL-8B-Instruct : remplissage du JSON de l'`Extraction` (deuxième lecture)."""

    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    server_url: str = "http://127.0.0.1:8000/v1"  # llama-server, API compatible OpenAI
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
    # Colonne des côtés imprimés : un côté sous 1 % de la longueur caractéristique
    # (racine de la superficie) est un débris de cellule, pas une limite de parcelle.
    sides_fragment_ratio: float = 0.01
    # Une colonne constante n'est crue que si la superficie imprimée correspond au
    # polygone régulier : c'est la seule parcelle réelle à côtés tous égaux.
    sides_regular_tolerance_pct: float = 5.0
    # Deux bornes peuvent partager une abscisse sur une parcelle rectangulaire ; trois
    # supposerait trois bornes alignées, ce qui trahit plutôt une case recopiée.
    max_identical_coordinates: int = 3
    # Relecture ciblee du tableau de coordonnees : la page entiere reste lue pour tous
    # les champs, et le tableau est en plus relu seul pour ne pas perdre les X/Y.
    table_rescan: bool = True
    table_zoom: int = 3
    # La superficie juge toute la validation geometrique : quand la page entiere ne la
    # donne pas, on relit de pres la region ou elle se trouve. Grossissement plus fort
    # que pour le tableau : la region est petite et tient en quelques caracteres.
    area_rescan: bool = True
    area_zoom: int = 4
    # Regularite d'une colonne de coordonnees : part minimale pour qu'une longueur,
    # un nombre de decimales ou un prefixe soit considere comme dominant.
    column_min_share: float = 0.75
    # Chevauchement : part de la plus petite parcelle recouverte, en pourcentage.
    # En deca, c'est une limite partagee entre voisins, pas un conflit : deux leves
    # de la meme limite different de quelques centimetres.
    overlap_min_share_pct: float = 1.0
    # Au-dela, une parcelle est consideree comme contenue dans l'autre.
    overlap_inclusion_pct: float = 95.0
    # Distance maximale entre une parcelle et la localite qu'elle declare, en km.
    # Une region est vaste, une commune ne l'est pas : le seuil suit le niveau.
    locality_commune_km: float = 25.0
    locality_department_km: float = 60.0
    locality_region_km: float = 150.0
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
    # Deux lectures qui ne diffèrent que par l'orthographe (« Mornoz »/« Mermoz »,
    # « 14 DEC 2020 »/« 1.4.DEC.2020 ») : accord, mais un cran sous l'accord exact.
    conf_near_agree: float = 0.95
    # Similarité minimale pour y prétendre. Les chiffres, eux, doivent être identiques.
    near_agree_ratio: float = 0.85
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
        server_kind=env.get("CADASTRE_PADDLE_SERVER_KIND", s.paddle.server_kind),
        server_url=env.get("CADASTRE_PADDLE_URL", s.paddle.server_url),
        device=env.get("CADASTRE_PADDLE_DEVICE", s.paddle.device),
    )
    qwen = replace(
        s.qwen,
        server_url=env.get("CADASTRE_QWEN_URL", s.qwen.server_url),
        model=env.get("CADASTRE_QWEN_MODEL", s.qwen.model),
        api_key=env.get("CADASTRE_QWEN_API_KEY", s.qwen.api_key),
    )
    return replace(s, paddle=paddle, qwen=qwen)


SETTINGS = Settings()
