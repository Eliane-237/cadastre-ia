# cadastre-ia

Modèle d'extraction des titres fonciers et plans de bornage (coordonnées UTM 28N) pour le
cadastre sénégalais.

Le système est un **outil d'alerte et d'aide à la décision**. Il extrait les champs de
l'`Extraction` (n° de titre, NICAD, propriétaire, localité, superficie, points X/Y, distances),
vérifie leur cohérence et dit ce qui doit être relu. Le rejet d'un dossier reste une décision
humaine et juridique.

## Architecture

```
document (photo, scan, PDF)
  │
  ├─ model/imaging.py      perspective (photo) et éclairage inégal, seulement si détectés
  │
  ├─ PaddleOCR-VL-1.6      orientation 90/180/270°, blocs de page, tableaux en HTML
  │  model/readers/paddle_vl.py + model/tables.py  → tableau de coordonnées (+ n° TF, NICAD,
  │                                                  superficie repérés par motif)
  │
  ├─ Qwen3-VL-8B-Instruct  JSON imposé au décodage : tous les champs, tableau compris
  │  model/readers/qwen_vl.py  → deuxième lecture indépendante, confiance par champ (logprobs)
  │
  ├─ model/fusion.py       compare les deux lectures, chiffre à chiffre
  │
  └─ model/validate.py     surface de Gauss, plage UTM 28N, côtés, polygone
                           → Extraction (+ issues, + provenance pour l'audit)
```

Les deux lecteurs ont des rôles différents :

- **PaddleOCR-VL-1.6** lit la page entière (titres, paragraphes, tableaux, cachets) et rend les
  tableaux en structure. Il ne sait pas quel texte est le numéro de titre : on s'en sert
  surtout pour lire **précisément le tableau de coordonnées**, là où un chiffre faux coûte cher.
- **Qwen3-VL-8B** suit le schéma qu'on lui impose : il remplit **tous les champs** en JSON, y
  compris le tableau, et sert de **deuxième lecture**. Il ne voit pas la sortie de PaddleOCR-VL :
  si c'était le cas, leurs erreurs seraient corrélées et leur accord ne prouverait plus rien.

### Règles de décision (model/fusion.py)

| situation | valeur retenue | source | confiance |
|---|---|---|---|
| les deux lecteurs lisent la même chaîne | cette chaîne | `consensus` | `conf_agree` (0,97) |
| un seul lecteur a lu une coordonnée ou une distance | sa lecture | `paddle_vl` / `qwen_vl` | 0,60 au plus → **revue humaine** |
| un seul lecteur a lu un champ texte (propriétaire, localité…) | sa lecture | `qwen_vl` le plus souvent | sa confiance (logprobs), plafonnée à 0,90 ; 0,60 s'il n'en donne pas |
| coordonnées divergentes, **une seule** combinaison cohérente avec la superficie et les côtés imprimés | cette combinaison | `geometry` | `conf_geometry` (0,90) |
| divergence que la géométrie ne départage pas (aucune ou plusieurs combinaisons cohérentes, ou pas de superficie ni de côté imprimés) | lecture de PaddleOCR-VL | `paddle_vl` | `conf_conflict` (0,30) → **revue humaine** |
| champ texte divergent (n° de titre, NICAD…) | lecture de Qwen3-VL | `qwen_vl` | `conf_conflict` → revue |

Rien n'est corrigé en silence : chaque valeur garde **toutes ses lectures** (`readings`), et
la fusion émet des alertes `READING_CONFLICT`, `READING_CONFLICT_RESOLVED`, `FIELD_CONFLICT`,
`POINT_COUNT_MISMATCH` et `READER_FAILED`. Un document part en revue dès qu'une valeur est sous
`review_threshold` (0,85) ou qu'une erreur est levée (`Extraction.needs_review()`).

Toutes ces confiances sont des **rangs, pas encore des probabilités**. Elles se calibrent sur
le jeu annoté avec `eval.metrics.calibration_table` et `threshold_for` (voir plus bas). Elles
sont toutes dans `model/config.py`.

## Installation

Poste de développement (tests, fusion, évaluation rejouée, client HTTP) :

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

Machine GPU (Linux ou WSL2, pilote NVIDIA, Docker avec nvidia-container-toolkit) :

```bash
# 1. Serveurs d'inférence des deux modèles (vLLM)
docker compose -f deploy/docker-compose.yml up -d

# 2. Client : PaddleOCR (mise en page + orientation) dans le processus Python
pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install -r requirements-gpu.txt

export CADASTRE_QWEN_URL=http://127.0.0.1:8000/v1
export CADASTRE_PADDLE_URL=http://127.0.0.1:8080/v1
```

`deploy/docker-compose.yml` explique comment répartir la mémoire sur un ou deux GPU. Sans vLLM,
`CADASTRE_QWEN_BACKEND=local` et `CADASTRE_PADDLE_BACKEND=local` chargent les modèles dans le
processus (plus lent, pas de JSON imposé au décodage pour Qwen : il est validé après coup).

Les documents ne quittent pas la machine : les deux serveurs sont locaux, aucune API externe
n'est appelée.

## Utilisation

```bash
python -m model data/raw/D01.jpg --out D01.json --readings D01_lectures.json
python -m model data/raw/D01.jpg --only paddle      # diagnostic d'un seul lecteur
```

La sortie standard reçoit l'`Extraction` en JSON. Le résumé (alertes, champs à relire) est
écrit sur stderr.

```python
from model.extract import extract

ex = extract(Path("data/raw/D01.jpg"), "D01")
ex.points[0].x            # Extracted[float] : value, confidence, source, readings
ex.issues                 # alertes expliquées
ex.needs_review()         # faut-il une relecture humaine ?
ex.provenance             # sha256 du document, modèles, prétraitement, durées, erreurs
```

## Évaluation

Les données réelles ne sont jamais versionnées (`data/raw/`, `data/annotations/`,
`reports/readings/` sont ignorés par git).

```bash
# Annotations depuis le classeur d'analyse
python -m eval.xlsx_to_annotations CLASSEUR.xlsx data/annotations

# Évaluation avec les modèles (GPU) : PaddleOCR-VL seul, Qwen3-VL seul, puis la fusion
python -m eval.run_baseline data/raw data/annotations reports

# Rejouer fusion + validation sur les lectures enregistrées (sans GPU) : utile pour régler
# les seuils de model/config.py
python -m eval.run_baseline data/raw data/annotations reports --replay

# Premier essai de la chaîne GPU sans données réelles : documents synthétiques fictifs
python -m eval.synth --out data/synth --n 20 --hard
python -m eval.run_baseline data/synth/raw data/synth/annotations reports/synth
```

Le rapport donne, pour chaque lecteur et pour la fusion, les valeurs exactes, presque justes
(un caractère), fausses et manquantes. Il donne aussi la calibration de la fusion (taux
d'erreur par tranche de confiance) et le seuil qui garantit au plus 1 % d'erreurs parmi les
valeurs acceptées sans relecture. Pour référence, RapidOCR seul donnait 52 % de valeurs exactes
sur 6 documents réels (`reports/baseline_rapidocr.json`).

## Arborescence

```
model/
  config.py          seuils, confiances, points d'accès des modèles (surcharge par l'environnement)
  schema.py          contrat de sortie : Extraction, Extracted (+ readings), Issue, Provenance
  annotation.py      format de la vérité terrain
  imaging.py         chargement image/PDF, perspective, éclairage
  numbers.py         normalisation des coordonnées, superficies, n° de titre, NICAD
  tables.py          tableau HTML → points et côtés
  readers/           paddle_vl.py, qwen_vl.py, prompts.py (schéma imposé), json_spans.py (logprobs)
  fusion.py          consensus, arbitrage géométrique, revue
  validate.py        cohérence géométrique
  extract.py         orchestration ; __main__.py : ligne de commande
eval/                métriques, évaluation, conversion du classeur, documents synthétiques
deploy/              serveurs vLLM (docker compose)
```

## Limites connues

- Pas encore mesuré sur les vrais titres avec les deux modèles : c'est la prochaine étape
  (`eval.run_baseline` sur le jeu annoté). Les confiances restent à calibrer.
- Un seul tableau de coordonnées par document. Plans multi-parcelles non gérés.
- La position exacte d'une cellule n'est pas fournie par PaddleOCR-VL : la `bbox` d'un point
  est celle du tableau entier.
- Un désaccord d'un centimètre sur une décimale reste souvent indécidable par la géométrie
  (la superficie et les côtés ne bougent presque pas). Il part en revue, par prudence.
- Si les deux lecteurs commettent la même erreur, le consensus ne la détecte pas. Seule la
  validation géométrique (superficie, côtés) peut encore la signaler.
