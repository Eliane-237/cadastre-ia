# Télécharge les modèles GGUF officiels dans models\ (une seule fois).
#   .\scripts\download_models.ps1                  # Qwen3-VL-8B en Q4_K_M (~5,8 Go)
#   .\scripts\download_models.ps1 -QwenQuant Q8_0  # Qwen3-VL-8B en Q8_0  (~9,5 Go)
param(
    [ValidateSet("Q4_K_M", "Q8_0")] [string] $QwenQuant = "Q4_K_M",
    [string] $Dest = (Join-Path $PSScriptRoot "..\models")
)
$ErrorActionPreference = "Stop"

# PaddleOCR-VL-1.6 (0,9 B) : ~1,8 Go
hf download PaddlePaddle/PaddleOCR-VL-1.6-GGUF `
    PaddleOCR-VL-1.6-GGUF.gguf PaddleOCR-VL-1.6-GGUF-mmproj.gguf `
    --local-dir (Join-Path $Dest "paddleocr-vl-1.6")

# Qwen3-VL-8B-Instruct : modèle de langue + encodeur d'image (mmproj)
hf download Qwen/Qwen3-VL-8B-Instruct-GGUF `
    "Qwen3VL-8B-Instruct-$QwenQuant.gguf" mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf `
    --local-dir (Join-Path $Dest "qwen3-vl-8b")
