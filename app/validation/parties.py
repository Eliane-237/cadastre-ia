"""Rapprocher deux écritures d'un même propriétaire, d'un titre à l'autre.

Le même requérant ne s'écrit pas deux fois pareil : « SOCIETE IMMOBILIERE DU CAP-VERT »
sur un titre, « Sté Immobilière du Cap Vert » sur le suivant, « DIOP Mamadou » ici et
« Mamadou DIOP » là. Tant que ces écritures restent étrangères l'une à l'autre, deux
parcelles du même propriétaire ne se voient pas, et un contrôle de cohérence élémentaire
— le cédant est-il bien le propriétaire enregistré ? — n'a rien sur quoi s'appuyer.

Trois choses que ce module ne fait pas, et c'est délibéré :

- il ne fusionne rien. Il propose un rapprochement et le donne à relire. **Un nom n'est
  pas une identité** : « Mamadou Diop » désigne des milliers de personnes au Sénégal, et
  décider que deux titres appartiennent au même homme sur la foi d'une chaîne de
  caractères produirait exactement le genre d'erreur qu'on ne rattrape pas ;
- il ne parle pas de fraude. Deux parcelles d'un même propriétaire, c'est le cas le plus
  banal du cadastre ;
- il ne remplace pas un identifiant. Le rapprochement sûr passe par le NIN ou le NINEA,
  que nos documents n'impriment pas et que le schéma n'extrait pas encore. Tant que
  c'est le cas, ce qui sort d'ici reste une piste pour un agent.

La comparaison se fait sur des jetons, pas sur la chaîne entière : c'est ce qui absorbe
l'inversion nom/prénom, la plus fréquente de nos variantes, et ce qu'un rapport de
similarité brut rate complètement.

Deux usages, à ne jamais confondre :

- `rapprocher()` cherche, dans toute la base, les écritures voisines d'un nom. C'est une
  aide à la recherche. Elle ne prouve rien et rend volontiers plusieurs titres : des
  milliers de personnes portent le même nom au Sénégal.
- `concordance()` vérifie. Elle part d'un lien **déjà établi** entre deux titres — même
  NICAD, filiation déclarée, ou même terrain constaté — et dit si les identités
  s'accordent. C'est l'ordre qui compte : le lien d'abord, le nom pour le confirmer ou
  le contredire. Un nom ne lie jamais deux titres à lui seul.

Ce que ce module ne peut pas faire avec nos documents : vérifier qu'un *cédant* est bien
le propriétaire enregistré. Un plan de bornage imprime un « Requérant », pas une vente.
Le contrôle cédant/cessionnaire suppose de lire l'acte notarié — un autre document, un
autre module.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum

from app.core.config import SETTINGS, Settings
from app.core.normalize import canonical_nicad, canonical_text, canonical_title
from app.geospatial.parcel import Parcel

# Civilités et titres : ils disent comment s'adresser à quelqu'un, pas qui il est.
_CIVILITES = frozenset({
    "m", "mr", "mme", "mlle", "monsieur", "madame", "mademoiselle", "dr", "pr",
    "me", "el", "hadj", "elhadj", "alhaji", "serigne", "cheikh", "mame",
})

# Mots-outils : présents ou absents selon le greffier, jamais distinctifs.
_OUTILS = frozenset({"de", "du", "des", "la", "le", "les", "et", "d", "l", "au", "aux", "a"})

# Formes abrégées courantes sur les titres sénégalais. On développe plutôt que
# d'écarter : « STE » et « SOCIETE » doivent se rejoindre, pas disparaître tous les deux.
_ABREVIATIONS = {
    "ste": "societe", "sté": "societe", "sté.": "societe", "soc": "societe",
    "cie": "compagnie", "ets": "etablissements", "etbs": "etablissements",
    "sarl": "sarl", "sa": "sa", "sci": "sci", "gie": "gie",
    "immob": "immobiliere", "imm": "immobiliere",
    "snc": "snc", "succ": "succession", "hrs": "heritiers", "her": "heritiers",
}

# Deux jetons se répondent au-dessus de ce rapport. Il est volontairement haut : il
# absorbe les variantes d'écriture d'un même mot — « héritiers »/« héritier » (0,94),
# « immobilière »/« immobilier » (0,95) — et rien d'autre.
#
# Il ne rattrape PAS les coquilles d'OCR sur un patronyme, et c'est délibéré :
# « diop »/« diof » et « sall »/« fall » donnent tous deux 0,750. Le seuil qui
# repêcherait la coquille confondrait SALL et FALL, deux familles sans rapport. Un nom
# mal lu est un défaut d'extraction — deux lecteurs et la réparation de caractères s'en
# chargent en amont — pas quelque chose à rattraper en rapprochant des identités.
_SEUIL_JETON = 0.85

# Formes juridiques : leur présence signale une personne morale, pour laquelle la
# convention du patronyme en capitales n'a pas cours.
_FORMES_JURIDIQUES = frozenset({
    "societe", "sci", "sa", "sarl", "gie", "snc", "compagnie", "etablissements",
    "succession", "heritiers", "cooperative", "association", "entreprise",
})

# « S.C.I. », « G.I.E. », « S.A. » : des sigles que la ponctuation disperserait en
# lettres isolées, aussitôt écartées comme trop courtes.
_SIGLE = re.compile(r"(?<![^\W\d_])((?:[^\W\d_]\s*\.){2,})")


class Niveau(str, Enum):
    PROBABLE = "probable"        # presque sûrement la même partie, à confirmer
    A_VERIFIER = "a_verifier"    # ressemblance notable, un agent doit trancher


@dataclass(frozen=True)
class Rapprochement:
    """Deux écritures qui pourraient désigner la même partie. Jamais une conclusion."""

    niveau: Niveau
    nom: str                 # tel qu'il est écrit sur le document en cours
    nom_voisin: str          # tel qu'il est écrit sur l'autre document
    document: str            # le document où se trouve l'autre écriture
    score: float             # 0 à 1

    @property
    def message(self) -> str:
        certitude = ("écritures très proches" if self.niveau is Niveau.PROBABLE
                     else "écritures voisines")
        return (f"« {self.nom} » et « {self.nom_voisin} » ({self.document}) : "
                f"{certitude} ({self.score:.0%}). À confirmer par un agent — "
                "un nom ne suffit pas à établir une identité.")


def jetons(nom: str | None) -> tuple[str, ...]:
    """Les mots distinctifs d'un nom, forme canonique, ordre indifférent.

    « M. Mamadou DIOP » et « Diop, Mamadou » rendent tous deux ('diop', 'mamadou') —
    triés, parce que l'ordre nom/prénom change d'un greffier à l'autre.
    """
    if not nom:
        return ()
    # Les sigles d'abord : « S.C.I. » doit devenir « sci », pas « s », « c », « i ».
    base = canonical_text(_SIGLE.sub(lambda m: m.group(1).replace(".", "").replace(" ", ""),
                                     nom)) if nom else None
    if not base:
        return ()
    mots = []
    for brut in base.replace("-", " ").replace("'", " ").replace("/", " ").split():
        mot = _ABREVIATIONS.get(brut, brut)
        if mot in _CIVILITES or mot in _OUTILS or len(mot) < 2:
            continue
        mots.append(mot)
    return tuple(sorted(mots))


def patronyme(nom: str | None) -> tuple[str, ...]:
    """Le ou les jetons écrits en capitales, quand le document distingue les casses.

    Les titres sénégalais écrivent le patronyme en capitales et le prénom en bas de
    casse : « DIOP Mamadou », « Mamadou DIOP ». C'est la casse qui porte l'information,
    pas la position — et la position change d'un greffier à l'autre.

    Rend un tuple vide quand il n'y a rien à lire : nom tout en capitales (l'OCR en
    produit souvent), tout en bas de casse, ou personne morale, dont le nom ne suit pas
    cette convention. Un signal absent ne se devine pas.
    """
    if not nom:
        return ()
    brut = _SIGLE.sub(lambda m: m.group(1).replace(".", "").replace(" ", ""), nom)

    utiles: list[tuple[str, str]] = []
    for mot in re.split(r"[\s,\-'’/]+", brut):
        canon = canonical_text(mot)
        if not canon:
            continue
        canon = _ABREVIATIONS.get(canon, canon)
        if canon in _CIVILITES or canon in _OUTILS or len(canon) < 2:
            continue
        utiles.append((mot, canon))

    if any(canon in _FORMES_JURIDIQUES for _, canon in utiles):
        return ()
    capitales = [canon for mot, canon in utiles if mot.isupper() and len(mot) > 1]
    # Tout en capitales, ou rien en capitales : le document ne distingue pas, on ne
    # distingue pas non plus.
    if not capitales or len(capitales) == len(utiles):
        return ()
    return tuple(sorted(capitales))


def similarite(a: str | None, b: str | None) -> float:
    """Part des jetons communs aux deux noms, coquilles d'OCR tolérées (0 à 1).

    Coefficient de Dice sur des jetons appariés : 2·m/(|A|+|B|), où m est le nombre de
    jetons de A qui trouvent un répondant dans B. « Mamadou Lamine DIOP » et
    « Mamadou DIOP » tombent à 0,80 — assez pour être proposés à un agent, pas pour être
    affirmés. Deux noms de deux mots doivent donc s'accorder sur les deux : « Fatou
    SALL » et « Fatou FALL » restent à 0,50, et ne sortent pas d'ici.
    """
    ja, jb = jetons(a), jetons(b)
    if not ja or not jb:
        return 0.0

    # Quand les deux documents écrivent le patronyme en capitales et qu'ils ne
    # désignent pas le même, il n'y a rien à rapprocher : « Mamadou DIOP » et
    # « MAMADOU Diop » partagent leurs deux mots et sont deux personnes distinctes.
    # La casse ne sert qu'à écarter, jamais à rapprocher : un signal faible peut
    # refuser un rapprochement, il ne doit pas en fabriquer un.
    pa, pb = patronyme(a), patronyme(b)
    if pa and pb and not _se_repondent(pa, pb):
        return 0.0

    restants = list(jb)
    apparies = 0
    for jeton in ja:
        meilleur, rapport = None, 0.0
        for candidat in restants:
            r = SequenceMatcher(None, jeton, candidat).ratio()
            if r > rapport:
                meilleur, rapport = candidat, r
        if meilleur is not None and rapport >= _SEUIL_JETON:
            restants.remove(meilleur)
            apparies += 1
    return 2 * apparies / (len(ja) + len(jb))


def _se_repondent(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Deux ensembles de jetons partagent-ils au moins un mot ?"""
    return bool(set(a) & set(b))


def rapprocher(nom: str | None, candidats: Iterable[tuple[str, str | None]],
               cfg: Settings = SETTINGS) -> list[Rapprochement]:
    """Les écritures de la base qui pourraient désigner la même partie que `nom`.

    Aide à la recherche, pas vérification : plusieurs titres peuvent sortir, et c'est
    normal — beaucoup de gens portent le même nom. Pour savoir si deux titres précis
    désignent la même partie, il faut un lien, et c'est `concordance()` qui le juge.

    `candidats` livre des couples (document_id, nom tel qu'écrit). Comme pour la
    confrontation géométrique, ce module ne va pas chercher la base lui-même : on lui
    donne les candidats, il juge. Les meilleurs scores d'abord.
    """
    if not jetons(nom):
        return []

    trouves: list[Rapprochement] = []
    for document, voisin in candidats:
        score = similarite(nom, voisin)
        if score >= cfg.party_match_pct:
            niveau = Niveau.PROBABLE
        elif score >= cfg.party_review_pct:
            niveau = Niveau.A_VERIFIER
        else:
            continue
        # `nom` est non vide : `jetons(nom)` l'a déjà établi.
        trouves.append(Rapprochement(niveau=niveau, nom=str(nom), nom_voisin=str(voisin),
                                     document=document, score=score))
    trouves.sort(key=lambda r: -r.score)
    return trouves


# --------------------------------------------------------------- vérification

class Lien(str, Enum):
    """Ce qui rattache deux titres l'un à l'autre, avant toute question d'identité."""

    NICAD = "nicad"              # même identifiant cadastral : la même parcelle
    COPROPRIETE = "copropriete"  # même parcelle d'assise, deux lots distincts
    TITRE_MERE = "titre_mere"    # filiation déclarée par l'un des deux documents
    GEOMETRIE = "geometrie"      # même terrain, constaté par la géométrie


# « Lot n° 12 », « LOT 12 », « lot:12 » désignent le même lot.
_PREFIXE_LOT = re.compile(r"^\s*lots?\s*(?:n\s*[°ºo]?)?\s*[:.\-]?\s*", re.IGNORECASE)


def _lot(valeur: str | None) -> str | None:
    """Le numéro de lot seul, sans le mot qui le précède."""
    if not valeur:
        return None
    return canonical_text(_PREFIXE_LOT.sub("", str(valeur)))


class Identite(str, Enum):
    CONCORDANTE = "concordante"
    DIVERGENTE = "divergente"
    INDETERMINEE = "indeterminee"    # un des deux noms manque ou reste ambigu


@dataclass(frozen=True)
class Concordance:
    """Deux titres liés, et ce que leurs requérants en disent."""

    lien: Lien
    identite: Identite
    document: str
    nom: str | None
    nom_voisin: str | None
    score: float
    reserve: str | None = None      # ce qui manque pour conclure, dit à l'agent

    @property
    def a_instruire(self) -> bool:
        """Deux requérants différents sur une même parcelle. Rien de plus, rien de moins.

        La filiation en est exclue à dessein : un morcellement vendu à un tiers porte
        normalement un autre nom que le titre mère. C'est le cas courant, pas l'anomalie.
        """
        return (self.identite is Identite.DIVERGENTE
                and self.lien in (Lien.NICAD, Lien.GEOMETRIE))

    @property
    def meme_parcelle(self) -> bool:
        """Les deux titres portent-ils sur la même emprise, au sens du décret ?"""
        return self.lien in (Lien.NICAD, Lien.GEOMETRIE)

    @property
    def message(self) -> str:
        par_quoi = {
            Lien.NICAD: "portent le même NICAD",
            Lien.COPROPRIETE: "partagent la parcelle d'assise d'une copropriété",
            Lien.TITRE_MERE: "sont liés par une filiation déclarée",
            Lien.GEOMETRIE: "occupent le même terrain",
        }[self.lien]
        entete = f"Ce titre et {self.document} {par_quoi}"
        if self.identite is Identite.INDETERMINEE:
            manquant = "ce document" if not self.nom else self.document
            return (f"{entete}, mais le requérant n'a pas été lu sur {manquant} : "
                    "l'identité ne peut pas être vérifiée.")
        if self.identite is Identite.CONCORDANTE:
            return (f"{entete}, et le requérant s'y écrit de la même façon "
                    f"({self.score:.0%}) : identités concordantes.")
        if self.lien is Lien.TITRE_MERE:
            return (f"{entete} et portent deux requérants différents "
                    f"(« {self.nom} » / « {self.nom_voisin} ») : "
                    "normal pour un morcellement cédé à un tiers.")
        if self.lien is Lien.COPROPRIETE:
            return (f"{entete} sous deux numéros de lot distincts, avec deux requérants "
                    f"différents (« {self.nom} » / « {self.nom_voisin} ») : "
                    "normal en copropriété.")
        suite = (f" {self.reserve}" if self.reserve else "")
        return (f"{entete} et portent deux requérants différents "
                f"(« {self.nom} » / « {self.nom_voisin} ») : à instruire.{suite}")


def lien(a: Parcel, b: Parcel) -> Lien | None:
    """Ce qui rattache `a` à `b`, du plus sûr au plus faible. None s'ils sont étrangers.

    Le NICAD est unique par parcelle (décret sur l'identification cadastrale, art. 1 et
    2), et un morcellement, une fusion ou toute modification de limites annule les
    anciens au profit de nouveaux (art. 4). Deux titres vivants ne devraient donc jamais
    le partager — sauf en copropriété, où il désigne la parcelle d'assise commune et où
    le numéro de lot distingue les biens (art. 6).

    Réserve : l'historique des NICAD annulés est conservé indéfiniment (art. 4, 2°). Un
    document ancien peut donc porter un NICAD qui ne désigne plus la même emprise, et
    rien dans nos documents ne dit qu'il a été annulé.

    La géométrie n'est pas regardée ici : elle est constatée par `overlap.scan`, qui
    mesure le recouvrement et son incertitude. On ne refait pas ce travail.
    """
    na, nb = canonical_nicad(a.nicad), canonical_nicad(b.nicad)
    if na and nb and na == nb:
        # Décret sur l'identification cadastrale, art. 6 : en copropriété, le bien est
        # identifié par son numéro de lot, qui s'ajoute au NICAD de la parcelle
        # d'assise. Deux lots d'un même immeuble partagent donc légitimement le NICAD,
        # et deux requérants différents n'y sont pas une anomalie.
        la, lb = _lot(a.lot), _lot(b.lot)
        if la and lb and la != lb:
            return Lien.COPROPRIETE
        return Lien.NICAD
    ta, tb = canonical_title(a.title_number), canonical_title(b.title_number)
    ma, mb = canonical_title(a.mother_title), canonical_title(b.mother_title)
    if (ma and tb and ma == tb) or (mb and ta and mb == ta):
        return Lien.TITRE_MERE
    return None


def concordance(a: Parcel, b: Parcel, nature: Lien | None = None,
                cfg: Settings = SETTINGS) -> Concordance | None:
    """Les identités de deux titres déjà liés s'accordent-elles ?

    `nature` force le lien quand il vient d'ailleurs — un recouvrement géométrique
    constaté par `overlap.scan`, par exemple. Sans lien, il n'y a rien à vérifier et
    la fonction rend None : deux noms qui se ressemblent ne font pas un dossier.
    """
    nature = nature or lien(a, b)
    if nature is None:
        return None

    # Même NICAD sans numéro de lot des deux côtés : une copropriété ne peut pas être
    # écartée, et c'est par là qu'un agent doit commencer avant d'ouvrir un dossier.
    reserve = None
    if nature is Lien.NICAD and not (_lot(a.lot) and _lot(b.lot)):
        reserve = ("Le numéro de lot n'a pas été lu sur les deux titres : "
                   "une copropriété ne peut pas être écartée.")

    score = similarite(a.owner, b.owner)
    if not a.owner or not b.owner:
        identite = Identite.INDETERMINEE
    elif score >= cfg.party_match_pct:
        identite = Identite.CONCORDANTE
    elif score < cfg.party_review_pct:
        identite = Identite.DIVERGENTE
    else:
        identite = Identite.INDETERMINEE
    return Concordance(lien=nature, identite=identite, document=b.document_id,
                       nom=a.owner, nom_voisin=b.owner, score=score, reserve=reserve)
