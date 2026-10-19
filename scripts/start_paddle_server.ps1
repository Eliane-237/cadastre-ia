# Serveur llama.cpp de PaddleOCR-VL-1.6 (~2,5 Go de VRAM), port 8080. Laisser la fenêtre ouverte.
param(
    [string] $LlamaServer = "llama-server",
    [string] $Models = (Join-Path $PSScriptRoot "..\models"),
    [int] $Port = 8080
)
$dir = Join-Path $Models "paddleocr-vl-1.6"
& $LlamaServer `
    -m (Join-Path $dir "PaddleOCR-VL-1.6-GGUF.gguf") `
    --mmproj (Join-Path $dir "PaddleOCR-VL-1.6-GGUF-mmproj.gguf") `
    --host 127.0.0.1 --port $Port `
    --temp 0 -ngl 99 -c 8192 -np 1
