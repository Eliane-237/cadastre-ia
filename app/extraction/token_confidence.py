"""Confiance par champ à partir des probabilités des tokens générés (logprobs).

Un VLM ne donne pas de score par champ. Mais on connaît la probabilité de chaque token qu'il a
écrit : la confiance d'une valeur est la probabilité de son token le moins sûr. Pour une
coordonnée, c'est exactement le risque qui compte (un seul chiffre hésitant suffit).

Ce module repère la position de chaque valeur scalaire dans le texte JSON produit, puis la
rapproche des tokens qui la recouvrent.
"""

from __future__ import annotations

import math

Path = tuple[str | int, ...]


class _Scanner:
    def __init__(self, text: str) -> None:
        self.t = text
        self.i = 0
        self.spans: dict[Path, tuple[int, int]] = {}

    def ws(self) -> None:
        while self.i < len(self.t) and self.t[self.i] in " \t\r\n":
            self.i += 1

    def value(self, path: Path) -> None:
        self.ws()
        c = self.t[self.i]
        if c == "{":
            self.i += 1
            self.ws()
            if self.t[self.i] == "}":
                self.i += 1
                return
            while True:
                self.ws()
                start, end = self.string()
                key = self.t[start:end]
                self.ws()
                self.i += 1  # ':'
                self.value((*path, key))
                self.ws()
                if self.t[self.i] == ",":
                    self.i += 1
                    continue
                self.i += 1  # '}'
                return
        if c == "[":
            self.i += 1
            self.ws()
            if self.t[self.i] == "]":
                self.i += 1
                return
            k = 0
            while True:
                self.value((*path, k))
                k += 1
                self.ws()
                if self.t[self.i] == ",":
                    self.i += 1
                    continue
                self.i += 1  # ']'
                return
        if c == '"':
            self.spans[path] = self.string()
            return
        start = self.i
        while self.i < len(self.t) and self.t[self.i] not in ",}] \t\r\n":
            self.i += 1
        self.spans[path] = (start, self.i)

    def string(self) -> tuple[int, int]:
        """Renvoie la position du contenu (sans les guillemets)."""
        self.i += 1
        start = self.i
        while self.t[self.i] != '"':
            self.i += 2 if self.t[self.i] == "\\" else 1
        end = self.i
        self.i += 1
        return start, end


def value_spans(text: str) -> dict[Path, tuple[int, int]]:
    """Position (début, fin) de chaque valeur scalaire du JSON, indexée par son chemin."""
    s = _Scanner(text)
    try:
        s.value(())
    except IndexError:
        pass  # JSON tronqué : on garde ce qui a été repéré
    return s.spans


def value_confidences(
    text: str, tokens: list[tuple[str, float]], json_start: int = 0
) -> dict[Path, float]:
    """`tokens` : (texte du token, logprob) dans l'ordre, recomposant `text` ; le JSON commence
    à `json_start` (texte éventuel avant, ex. « ```json »). Renvoie la probabilité minimale des
    tokens recouvrant chaque valeur. Vide si les tokens ne recomposent pas le texte."""
    if not tokens or "".join(t for t, _ in tokens) != text:
        return {}
    starts, pos = [], 0
    for tok, _ in tokens:
        starts.append(pos)
        pos += len(tok)
    out: dict[Path, float] = {}
    spans = value_spans(text[json_start:]).items()
    for path, (a, b) in ((p, (a + json_start, b + json_start)) for p, (a, b) in spans):
        if b <= a:
            continue
        lps = [
            lp for (tok, lp), s in zip(tokens, starts, strict=True) if s < b and s + len(tok) > a
        ]
        if lps:
            out[path] = math.exp(min(lps))
    return out
