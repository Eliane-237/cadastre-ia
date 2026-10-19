# Ouvre les deux serveurs llama.cpp, chacun dans sa fenêtre (mode simple, Qwen en Q4_K_M).
#   .\scripts\start_servers.ps1
# Puis attendre que « python -m model.check_servers » affiche deux ✓.
param([string] $LlamaServer = "llama-server")
$here = $PSScriptRoot
Start-Process powershell -ArgumentList "-NoExit", "-File", "$here\start_paddle_server.ps1", "-LlamaServer", $LlamaServer
Start-Process powershell -ArgumentList "-NoExit", "-File", "$here\start_qwen_server.ps1", "-LlamaServer", $LlamaServer
Write-Host "Deux fenêtres ouvertes. Vérifier avec : python -m model.check_servers"
