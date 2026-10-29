"""Où se trouvent les localités sénégalaises, pour vérifier qu'un titre est chez lui.

La superficie et les longueurs de côtés vérifient la *forme* d'une parcelle ; rien ne
vérifiait sa *position*. Or une erreur sur les premiers chiffres d'une coordonnée — 
ceux qu'on ne voit jamais se tromper — déplacerait la parcelle de plusieurs kilomètres
sans changer son aire ni ses côtés d'un centimètre.

Les coordonnées viennent de GeoNames, via le paquet `geonamescache`, converties une
fois en UTM 28N : le contrôle se fait donc en mètres, sans reprojection et sans
dépendance au moment de l'exécution.

C'est la forme la plus simple de la première loi de Tobler appliquée au foncier :
une parcelle doit se trouver près de la commune qu'elle déclare.
"""

from __future__ import annotations

from app.core.normalize import canonical_text

# Nom canonique -> (Est, Nord) en EPSG:32628. Source : GeoNames (geonamescache).
LOCALITES: dict[str, tuple[float, float]] = {
    "bambey": (342904, 1625735),
    "bargny": (259855, 1626007),
    "bargny guedj": (260067, 1624888),
    "bargny ngoude": (259297, 1625681),
    "bignona": (366899, 1416471),
    "dagana": (435958, 1821758),
    "dakar": (236805, 1625873),
    "dara": (448488, 1696923),
    "diamniadio": (265110, 1628550),
    "diaoube": (590400, 1428068),
    "diourbel": (366075, 1619753),
    "dougnane": (298892, 1654590),
    "fatick": (348466, 1584944),
    "gandiaye": (362738, 1575045),
    "gossas village": (379972, 1601557),
    "goudomp": (405255, 1390602),
    "guediawaye": (241419, 1634763),
    "guinguineo": (398056, 1577739),
    "jaxaay parcelle niakoul rap": (254546, 1634658),
    "joal": (300413, 1567658),
    "kaffrine": (440545, 1559512),
    "kahone": (388223, 1565490),
    "kanel": (695638, 1713532),
    "kaolack": (384241, 1564798),
    "kayar": (271977, 1650446),
    "kbombole": (317017, 1633211),
    "kebemer": (346096, 1699208),
    "kedougou": (806370, 1389633),
    "keur medoune": (402665, 1631794),
    "khombole": (317919, 1632866),
    "kolda": (506374, 1425403),
    "koumpentoum": (546799, 1545925),
    "koungheul": (521600, 1545892),
    "koungneul sosse": (519801, 1544048),
    "le plateau": (237235, 1622383),
    "linguere": (487198, 1702052),
    "louga": (368750, 1727133),
    "matam": (686994, 1731639),
    "mbake": (402282, 1635332),
    "mboro": (297254, 1674739),
    "mbour": (288289, 1595298),
    "medina gounas": (634874, 1452516),
    "mekhe": (325713, 1671102),
    "ndibene dahra": (448842, 1695303),
    "ndioum": (537664, 1825707),
    "nguekhokh": (283985, 1605360),
    "niakoul rab": (255270, 1635698),
    "nioro du rip": (413512, 1520221),
    "n’diareme limamoulaye": (243368, 1635520),
    "ouro sogui": (679870, 1726049),
    "pikine": (242636, 1633656),
    "poukham un": (347704, 1587731),
    "pout": (278144, 1634014),
    "richard-toll": (425190, 1820225),
    "rufisque": (255222, 1628085),
    "rufisque est": (255278, 1628170),
    "saint-louis": (340622, 1771490),
    "sali": (282302, 1597469),
    "sebikhotane": (269851, 1631239),
    "sedhiou": (439534, 1404916),
    "tambacounda": (644068, 1522769),
    "thiadiaye": (316452, 1594898),
    "thies": (292703, 1635927),
    "thies nones": (288320, 1635289),
    "thilogne": (643672, 1760978),
    "tiebo": (367176, 1618133),
    "tivaouane": (304780, 1653800),
    "touba": (405833, 1643279),
    "velingara": (595735, 1453892),
    "ziguinchor": (361681, 1389699),
}


def position(nom: str | None) -> tuple[float, float] | None:
    """La position de la localité déclarée, ou None si elle nous est inconnue.

    Un nom inconnu n'est pas une anomalie : la base ne couvre que les localités
    principales. Le contrôle se tait plutôt que d'accuser.
    """
    if not nom:
        return None
    cle = canonical_text(nom)
    if cle and cle in LOCALITES:
        return LOCALITES[cle]
    # « Dakar, Médina » ou « Thiès Nord » : on essaie chaque partie du libellé.
    for morceau in (nom.replace(",", " ").split() if cle else []):
        part = canonical_text(morceau)
        if part and part in LOCALITES:
            return LOCALITES[part]
    return None


def distance_m(x: float, y: float, nom: str | None) -> float | None:
    """Distance en mètres entre un point UTM 28N et la localité déclarée."""
    repere = position(nom)
    if repere is None:
        return None
    return float(((x - repere[0]) ** 2 + (y - repere[1]) ** 2) ** 0.5)


def localite_connue(nom: str | None) -> bool:
    """Ce nom figure-t-il au répertoire ?

    Sert à départager deux lectures d'un champ de localité, jamais à rejeter une valeur :
    le répertoire ne couvre qu'une partie des localités sénégalaises, et beaucoup de
    communes réelles n'y sont pas.
    """
    canon = canonical_text(nom)
    return bool(canon) and canon in LOCALITES
