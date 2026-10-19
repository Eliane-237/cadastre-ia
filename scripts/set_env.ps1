# Réglages du projet pour la session PowerShell en cours :
#   . .\scripts\set_env.ps1
$env:CADASTRE_PADDLE_URL = "http://127.0.0.1:8080/v1"
$env:CADASTRE_QWEN_URL = "http://127.0.0.1:8000/v1"
# Mise en page PaddleOCR (PP-DocLayout) sur le processeur : toute la VRAM reste aux deux modèles.
$env:CADASTRE_PADDLE_DEVICE = "cpu" #"gpu:0"
Write-Host "Réglages cadastre-ia chargés."
