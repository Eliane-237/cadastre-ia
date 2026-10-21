"""Journal des décisions humaines, en append-only.

Le système constate ; l'humain tranche. Ce qu'il a tranché doit rester : pour revenir en
arrière, pour expliquer à un requérant pourquoi son titre a été bloqué, et pour qu'une
décision ne puisse pas être réécrite après coup. D'où l'écriture seule, une ligne JSON
par décision, jamais de modification ni de suppression.

Ce fichier n'est pas une base : c'est une trace. Il grossit, et c'est voulu.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class Action(str, Enum):
    ACCEPTEE = "acceptee"            # l'agent valide l'extraction et l'enregistre
    REJETEE = "rejetee"              # l'agent la refuse
    CORRIGEE = "corrigee"            # l'agent a modifié une valeur avant de trancher
    CHEVAUCHEMENT_LEVE = "chevauchement_leve"   # confrontation examinée, conflit écarté
    COMMENTEE = "commentee"          # observation sans décision


@dataclass(frozen=True)
class Decision:
    """Une décision humaine, horodatée et attribuée."""

    document_id: str
    action: Action
    auteur: str
    motif: str = ""
    detail: dict | None = None
    horodatage: str = ""

    def __post_init__(self) -> None:
        if not self.horodatage:
            object.__setattr__(self, "horodatage",
                               datetime.now(timezone.utc).isoformat())


class Historique:
    """Un fichier JSONL : une décision par ligne, ajoutée et jamais reprise."""

    def __init__(self, chemin: Path | str = Path("data/historique.jsonl")) -> None:
        self.chemin = Path(chemin)
        self.chemin.parent.mkdir(parents=True, exist_ok=True)

    def consigne(self, decision: Decision) -> Decision:
        ligne = asdict(decision) | {"action": decision.action.value}
        with self.chemin.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
        return decision

    def pour(self, document_id: str) -> list[Decision]:
        """Toutes les décisions prises sur un document, dans l'ordre où elles l'ont été."""
        return [d for d in self.tout() if d.document_id == document_id]

    def tout(self) -> list[Decision]:
        if not self.chemin.exists():
            return []
        out = []
        for ligne in self.chemin.read_text(encoding="utf-8").splitlines():
            if not ligne.strip():
                continue
            try:
                brut = json.loads(ligne)
                out.append(Decision(
                    document_id=brut["document_id"],
                    action=Action(brut["action"]),
                    auteur=brut["auteur"],
                    motif=brut.get("motif", ""),
                    detail=brut.get("detail"),
                    horodatage=brut.get("horodatage", ""),
                ))
            except (ValueError, KeyError):
                continue          # une ligne abîmée n'efface pas les autres
        return out
