"""Normalisation des valeurs lues, pour comparer les lecteurs chiffre à chiffre.

Les coordonnées restent des chaînes jusqu'au bout de la comparaison : « 235906.620 » et
« 235906.62 » ne sont pas la même lecture (le nombre de décimales fait partie de la vérité),
et convertir trop tôt en float masquerait des désaccords.
"""

from __future__ import annotations

import re
import unicodedata

from model.schema import Area

# Confusions lettre/chiffre fréquentes dans les zones numériques.
_DIGIT_CONFUSIONS = str.maketrans({"O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "|": "1"})
_COORD = re.compile(r"^\d{5,8}(?:\.\d{1,3})?$")
_LENGTH = re.compile(r"^\d{1,5}(?:\.\d{1,3})?$")


def _fix_token(token: str) -> str:
    digits = sum(c.isdigit() for c in token)
    letters = [c for c in token if c.isalpha()]
    if digits and len(letters) <= digits and all(c in "OoDlI|" for c in letters):
        return token.translate(_DIGIT_CONFUSIONS)
    return token


def fix_digits(text: str) -> str:
    """'l25,3O' → '125,30' dans les jetons numériques ; '1234/DK' ou 'B1' restent intacts.
    Normalise aussi la pleine chasse et les espaces insécables (NFKC)."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\S+", lambda m: _fix_token(m.group()), text)


def canonical_number(raw: str | None) -> str | None:
    """Forme canonique d'un nombre imprimé, décimales conservées telles quelles.

    '1 564 042,96' → '1564042.96' ; '381.210,57' → '381210.57' ; '235906.620' → '235906.620'.
    """
    if raw is None:
        return None
    s = fix_digits(str(raw)).strip()
    s = re.sub(r"\s*(?:m²|m2|m)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", "", s)
    if not re.fullmatch(r"[\d.,]+", s) or not re.search(r"\d", s):
        return None
    last_dot, last_comma = s.rfind("."), s.rfind(",")
    if last_dot >= 0 and last_comma >= 0:
        dec = "," if last_comma > last_dot else "."
        integer, decimals = s.rsplit(dec, 1)
        integer = re.sub(r"[.,]", "", integer)
    elif last_comma >= 0 or last_dot >= 0:
        sep = "," if last_comma >= 0 else "."
        parts = s.split(sep)
        # Un seul séparateur suivi de 1 à 3 chiffres : séparateur décimal. Plusieurs : milliers.
        if len(parts) == 2 and 1 <= len(parts[1]) <= 3 and parts[0]:
            integer, decimals = parts
        elif all(len(p) == 3 for p in parts[1:]) and parts[0]:
            integer, decimals = "".join(parts), ""
        else:
            return None
    else:
        integer, decimals = s, ""
    if not integer.isdigit() or (decimals and not decimals.isdigit()):
        return None
    integer = integer.lstrip("0") or "0"
    return f"{integer}.{decimals}" if decimals else integer


def canonical_coordinate(raw: str | None) -> str | None:
    """Comme `canonical_number`, restreint à la forme d'une coordonnée UTM (5 à 8 chiffres)."""
    c = canonical_number(raw)
    return c if c is not None and _COORD.match(c) else None


def canonical_length(raw: str | None) -> str | None:
    c = canonical_number(raw)
    return c if c is not None and _LENGTH.match(c) else None


def integer_digits(canonical: str) -> int:
    return len(canonical.split(".")[0])


# --------------------------------------------------------------------------- superficie

# Les unités sont imprimées en exposant. Selon la région lue, l'OCR rend « ᵃ » par
# « ° », « o », « º », une apostrophe ou une étoile, et « ᶜᵃ » par une apostrophe
# suivie ou non de « ca ».
_ARES_MARK = r"[a°oº'’\"*]"
_SEP_UNITE = r"[\s.]*"          # « 8.a 07ca » : un point separe parfois le nombre de l'unite
_AREA_HA_A_CA = re.compile(
    r"(?:(\d{1,4})" + _SEP_UNITE + r"ha(?![a-z]))?"
    r"\s*(?:(\d{1,2})" + _SEP_UNITE + _ARES_MARK + r"(?:res?)?(?![a-z]))?"
    r"\s*(?:(\d{1,2})" + _SEP_UNITE + r"(?:ca|['’\"]\s*ca|['’\"])(?![a-z]))?",
    re.IGNORECASE,
)
# Les centiares suivent toujours les ares : quand leur exposant a disparu, un groupe
# nu de deux chiffres juste après les ares est leur place (« 03ᵃ 00 » → 3 a 0 ca).
_CA_NUE = re.compile(r"\s*(\d{2})(?![\d.,])")
# Repli positionnel, quand plus aucun exposant n'a survécu à l'OCR. La contenance
# s'écrit toujours dans l'ordre ha, a, ca, les ares et centiares sur deux chiffres :
# deux groupes valent « a ca », trois valent « ha a ca ». Les séparateurs excluent
# « / », « - » et « . », pour qu'une date (« 25/09/24 ») ne devienne pas une surface.
_SEP_POSITIONNEL = r"[\s:;=]+"
_POSITIONNEL = re.compile(
    r"^[\s:;=]*(\d{2})" + _SEP_POSITIONNEL + r"(\d{2})"
    r"(?:" + _SEP_POSITIONNEL + r"(\d{2}))?\s*$"
)
# 'm?' : l'OCR lit souvent l'exposant ² comme un point d'interrogation.
_AREA_M2 = re.compile(r"(\d[\d\s.,]*\d|\d)\s*(?:m\s?[²2?]|m[eè]tres?\s+carr[ée]s)", re.IGNORECASE)


def _area_ou_none(ha: str | None, a: str | None, ca: str | None) -> Area | None:
    """Vérifie l'algèbre des unités avant de livrer la valeur.

    100 centiares font un are et 100 ares font un hectare : au-delà, la lecture est
    fausse, pas seulement inhabituelle. Mieux vaut refuser que livrer un juge menteur.
    """
    if int(a or 0) >= 100 or int(ca or 0) >= 100:
        return None
    return Area(ha=int(ha or 0), a=int(a or 0), ca=int(ca or 0))


def parse_area(text: str | None) -> Area | None:
    """'00 ha 06 a 49 ca' → Area(a=6, ca=49) ; '649 m²' → Area(a=6, ca=49).

    Rend None quand le texte est trop abîmé pour être lu : cette valeur sert de juge
    à toute la validation géométrique, une superficie inventée est pire qu'absente.
    """
    if not text:
        return None
    text = fix_digits(text)
    for m in _AREA_HA_A_CA.finditer(text):
        ha, a, ca = m.groups()
        if not any(v is not None for v in (ha, a, ca)):
            continue
        # Le premier chiffre retenu ne doit pas être collé à du bruit alphanumérique :
        # dans « 8607E1A 73Ca », le « 1A » n'est pas un are, c'est un débris.
        premier = next(i for i in (1, 2, 3) if m.group(i) is not None)
        debut = m.start(premier)
        if debut > 0 and (text[debut - 1].isalnum() or text[debut - 1] in ".,"):
            continue
        if ca is None and a is not None:
            if suite := _CA_NUE.match(text, m.end()):
                ca = suite.group(1)
        # Exiger au moins deux unités, ou « ha » seul : « 5 a » isolé est trop ambigu.
        if sum(v is not None for v in (ha, a, ca)) >= 2 or (ha and not (a or ca)):
            if (area := _area_ou_none(ha, a, ca)) is not None:
                return area
    m = _AREA_M2.search(text)
    if m:
        # `fix_digits` a déjà ramené « ² » à « 2 » : on cherche les nombres concurrents
        # autour du motif, jamais dans son suffixe d'unité. « Lot 59 : 1er 50m² » est
        # ambigu — trois nombres se disputent la place — donc illisible.
        autour = text[: m.start(1)] + text[m.end() :]
        value = canonical_number(m.group(1))
        if value is not None and not re.search(r"\d", autour):
            m2 = round(float(value))
            ha_, rest = divmod(m2, 10_000)
            return Area(ha=ha_, a=rest // 100, ca=rest % 100)
    if p := _POSITIONNEL.match(text):
        groupes = [g for g in p.groups() if g is not None]
        ha, a, ca = ("0", *groupes) if len(groupes) == 2 else tuple(groupes)
        return _area_ou_none(ha, a, ca)
    return None


# --------------------------------------------------------------------------- texte


def canonical_text(text: str | None) -> str | None:
    """Pour comparer deux lectures d'un nom ou d'une localité : sans accents ni casse."""
    if text is None:
        return None
    s = unicodedata.normalize("NFKD", text)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[\s.,;:]+", " ", s).strip().lower()
    return s or None


_TITLE_CORE = re.compile(r"(\d[\d .]*\d|\d)\s*/\s*([A-Za-z]{1,4})\b")


def canonical_title(text: str | None) -> str | None:
    """Noyau d'un numéro de titre : 'TF n° 12.345/DK' → '12345/DK'. Sinon texte normalisé."""
    if text is None:
        return None
    m = _TITLE_CORE.search(fix_digits(text))
    if m:
        return f"{int(re.sub(r'[ .]', '', m.group(1)))}/{m.group(2).upper()}"
    return canonical_text(text)


def canonical_nicad(text: str | None) -> str | None:
    if text is None:
        return None
    digits = re.sub(r"\D", "", fix_digits(text))
    return digits or None


def canonical_label(label: str | None) -> str | None:
    """'B 1', 'b-1', 'B01' → 'B1' ; '12' → '12'."""
    if label is None:
        return None
    s = re.sub(r"[\s\-_.]", "", fix_digits(str(label))).upper()
    m = re.fullmatch(r"([A-Z]*)0*(\d+)", s)
    return f"{m.group(1)}{m.group(2)}" if m else (s or None)
