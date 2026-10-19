# cadastre-ia

Extraction des titres fonciers et plans de bornage du cadastre sénégalais (coordonnées UTM 28N).

Le système lit un document (scan, photo, PDF) et remplit l'`Extraction` : n° de titre, NICAD,
propriétaire, localité, superficie, points X/Y, distances. Il vérifie ensuite la cohérence de
ces valeurs et signale ce qui doit être relu. C'est un **outil d'alerte et d'aide à la
décision** : le rejet d'un dossier reste une décision humaine et juridique.

## Démarrage rapide (Windows, GPU NVIDIA)

Prévu pour une RTX 3060 12 Go. Deux modèles tournent chacun derrière un serveur llama.cpp.

**1. Installer (une seule fois)**

- Python 3.11 et Git pour Windows.
- llama.cpp pour Windows CUDA : sur <https://github.com/ggml-org/llama.cpp/releases>,
  télécharger `llama-…-bin-win-cuda-12.4-x64.zip` et `cudart-llama-bin-win-cuda-12.4-x64.zip`,
  les dézipper dans un même dossier et ajouter ce dossier au `PATH`.
- Le projet, dans PowerShell à la racine du dépôt :

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install -r requirements-dev.txt
.\scripts\download_models.ps1
pytest
```

`pytest` doit afficher que tous les tests passent. Ils n'ont pas besoin des modèles.

**2. Lancer les deux serveurs**

```powershell
.\scripts\start_servers.ps1          # ouvre deux fenêtres (PaddleOCR-VL et Qwen3-VL)
. .\scripts\set_env.ps1
python -m model.check_servers        # attendre deux ✓
```

**3. Extraire un document**

```powershell
python -m model data\raw\D01.jpg --out D01.json
```

L'`Extraction` est écrite en JSON dans `D01.json`. Le résumé s'affiche à l'écran : alertes,
champs à relire, et si le document doit partir en revue humaine.

**4. Mesurer sur les documents annotés**

```powershell
python -m eval.xlsx_to_annotations C:\chemin\classeur.xlsx data\annotations
python -m eval.evaluate data\raw data\annotations reports
```

Sans documents réels, on peut d'abord tester sur des documents fictifs :

```powershell
python -m eval.synthetic_documents --out data\synth --n 10
python -m eval.evaluate data\synth\raw data\synth\annotations reports\synth
```

## Comment ça marche

```
document ─► preprocess.py        perspective d'une photo, éclairage inégal (si détectés)
         ─► paddle_reader.py     PaddleOCR-VL-1.6 : orientation, blocs de la page,
                                 tableau de coordonnées en structure (coordinate_table.py)
         ─► qwen_reader.py       Qwen3-VL-8B : tous les champs en JSON imposé (qwen_prompt.py),
                                 confiance par champ (token_confidence.py)
         ─► merge_readings.py    compare les deux lectures chiffre à chiffre
         ─► validate.py          surface de Gauss, plage UTM 28N, côtés, polygone
         ─► Extraction           valeurs + confiance + toutes les lectures + alertes
```

- **PaddleOCR-VL** lit la page entière et rend les tableaux en lignes et colonnes. On s'en sert
  surtout pour lire **le tableau de coordonnées**, là où un chiffre faux coûte cher.
- **Qwen3-VL** suit le schéma JSON qu'on lui impose. Il remplit **tous les champs** et sert de
  **deuxième lecture**. Il ne voit pas ce qu'a lu PaddleOCR-VL : leurs erreurs restent
  indépendantes, donc leur accord est une vraie preuve.

### Comment une valeur est retenue (merge_readings.py)

| situation | valeur retenue | confiance |
|---|---|---|
| les deux modèles lisent la même chaîne | cette chaîne (`consensus`) | 0,97 |
| coordonnées différentes, mais **une seule** combinaison cohérente avec la superficie et les côtés imprimés | cette combinaison (`geometry`) | 0,90 |
| différence que la géométrie ne départage pas | lecture de PaddleOCR-VL | 0,30 → **revue humaine** |
| coordonnée lue par un seul modèle | sa lecture | 0,60 au plus → **revue humaine** |
| champ texte lu par un seul modèle (propriétaire…) | sa lecture | sa confiance, 0,90 au plus |

Un document part en revue dès qu'une valeur est sous 0,85 ou qu'une erreur est détectée.
Rien n'est corrigé en silence : chaque valeur garde toutes ses lectures (`readings`). Ces
confiances sont des valeurs de départ, à calibrer sur les documents annotés (le rapport
d'`eval.evaluate` donne le taux d'erreur par tranche de confiance). Elles se règlent toutes
dans `model/config.py`.

## Deux modes pour une carte de 12 Go

| mode | Qwen3-VL-8B | VRAM | quand l'utiliser |
|---|---|---|---|
| simple : les deux serveurs en même temps | Q4_K_M | ~10 Go | usage courant, un document à la fois |
| précis : deux passes, un serveur à la fois | Q8_0 | ~10,5 Go | évaluation, traitement par lots |

Le Q4 est plus compressé : Qwen peut hésiter sur un chiffre, et la fusion envoie alors
davantage de documents en revue. Mode précis :

```powershell
.\scripts\download_models.ps1 -QwenQuant Q8_0
. .\scripts\set_env.ps1

# Fenêtre 1 : .\scripts\start_paddle_server.ps1        Fenêtre 2 :
python -m eval.evaluate data\raw data\annotations reports --only paddle

# Fenêtre 1 : Ctrl+C, puis .\scripts\start_qwen_server.ps1 -Quant Q8_0        Fenêtre 2 :
python -m eval.evaluate data\raw data\annotations reports --only qwen
```

La seconde passe complète les lectures enregistrées (`reports/readings/`), reprend
l'orientation trouvée par PaddleOCR-VL et écrit le rapport. `--replay` rejoue la fusion sur
ces lectures sans relancer les modèles : pratique pour régler les seuils.

## Fichiers

```
model/
  config.py            seuils, confiances, adresses des serveurs
  schema.py            format de sortie : Extraction, Extracted (+ readings), Issue, Provenance
  annotation.py        format de la vérité terrain
  preprocess.py        chargement image/PDF, perspective, éclairage
  normalize.py         mise en forme comparable : coordonnées, superficies, n° de titre, NICAD
  coordinate_table.py  tableau HTML (PaddleOCR-VL) → points et côtés
  reading.py           ce que rend un lecteur, avant la fusion
  paddle_reader.py     lecteur PaddleOCR-VL-1.6
  qwen_reader.py       lecteur Qwen3-VL-8B ; qwen_prompt.py : consigne et schéma imposé
  token_confidence.py  confiance par champ à partir des probabilités des tokens
  merge_readings.py    fusion des deux lectures, arbitrage géométrique
  validate.py          contrôles géométriques
  extract.py           enchaîne le tout ; __main__.py : ligne de commande
  check_servers.py     vérifie que les deux serveurs répondent
eval/
  evaluate.py          mesure chaque lecteur et la fusion sur un jeu annoté
  metrics.py           exact / presque / faux / manquant, calibration des confiances
  xlsx_to_annotations.py, synthetic_documents.py
scripts/               téléchargement des modèles et lancement des serveurs (PowerShell)
```

Les données réelles et les lectures ne sont jamais versionnées (`data/raw/`, `data/annotations/`, `reports/readings/`
et `models/` sont ignorés par git). Les documents ne quittent pas la machine : les deux
serveurs tournent en local.

## Limites connues

- Pas encore mesuré sur les vrais titres avec ces deux modèles : c'est la prochaine étape.
  Pour référence, RapidOCR seul donnait 52 % de valeurs exactes sur 6 documents réels
  (`reports/baseline_rapidocr.json`).
- Un seul tableau de coordonnées par document.
- Un écart d'un centimètre sur une décimale est souvent indécidable par la géométrie : il part
  en revue, par prudence.
- Si les deux modèles commettent la même erreur, seule la validation géométrique peut encore la
  signaler.

powershell -ExecutionPolicy Bypass -File .\scripts\start_qwen_server.ps1 -Quant Q8_0
.\scripts\start_paddle_server.ps1  

Get-ChildItem data\raw\* | ForEach-Object {
  $b = $_.BaseName
  python -m model $_.FullName --out "reports\readings\$b.json" --readings "reports\readings\$($b)_readings.json"
}
python -m eval.evaluate data\raw data\annotations reports