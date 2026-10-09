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

_AREA_HA_A_CA = re.compile(
    # (?![a-z]) plutôt que \b : l'OCR colle souvent unité et chiffres ('01a63 ca').
    r"(?:(\d{1,4})\s*ha(?![a-z]))?\s*(?:(\d{1,2})\s*a(?:res?)?(?![a-z]))?"
    r"\s*(?:(\d{1,2})\s*ca(?![a-z]))?",
    re.IGNORECASE,
)
# 'm?' : l'OCR lit souvent l'exposant ² comme un point d'interrogation.
_AREA_M2 = re.compile(r"(\d[\d\s.,]*\d|\d)\s*(?:m\s?[²2?]|m[eè]tres?\s+carr[ée]s)", re.IGNORECASE)


def parse_area(text: str | None) -> Area | None:
    """'00 ha 06 a 49 ca' → Area(a=6, ca=49) ; '649 m²' → Area(a=6, ca=49)."""
    if not text:
        return None
    text = fix_digits(text)
    for m in _AREA_HA_A_CA.finditer(text):
        ha, a, ca = m.groups()
        # Exiger au moins deux unités, ou « ha » seul : « 5 a » isolé est trop ambigu.
        if sum(v is not None for v in (ha, a, ca)) >= 2 or (ha and not (a or ca)):
            return Area(ha=int(ha or 0), a=int(a or 0), ca=int(ca or 0))
    m = _AREA_M2.search(text)
    if m:
        value = canonical_number(m.group(1))
        if value is not None:
            m2 = round(float(value))
            ha, rest = divmod(m2, 10_000)
            return Area(ha=ha, a=rest // 100, ca=rest % 100)
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
