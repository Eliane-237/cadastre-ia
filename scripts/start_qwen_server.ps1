# Serveur llama.cpp de Qwen3-VL-8B, port 8000. Laisser la fenêtre ouverte.
#   Q4_K_M : ~7,5 Go de VRAM, tient à côté de PaddleOCR-VL sur une carte de 12 Go.
#   Q8_0   : ~10,5 Go, plus fidèle sur les chiffres ; sur 12 Go, à faire tourner seul
#            (lecture en deux passes, voir README).
# En cas de « out of memory » : -Context 6144.
param(
    [ValidateSet("Q4_K_M", "Q8_0")] [string] $Quant = "Q4_K_M",
    [string] $LlamaServer = "llama-server",
    [string] $Models = (Join-Path $PSScriptRoot "..\models"),
    [int] $Port = 8000,
    [int] $Context = 8192
)
$dir = Join-Path $Models "qwen3-vl-8b"
& $LlamaServer `
    -m (Join-Path $dir "Qwen3VL-8B-Instruct-$Quant.gguf") `
    --mmproj (Join-Path $dir "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf") `
    --alias Qwen/Qwen3-VL-8B-Instruct `
    --host 127.0.0.1 --port $Port `
    --temp 0 --jinja -ngl 99 -c $Context -np 1 `
    -fa on --cache-type-k q8_0 --cache-type-v q8_0
