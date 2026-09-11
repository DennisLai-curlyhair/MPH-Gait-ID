param(
    [ValidateSet("offline", "realtime")]
    [string]$Profile = "realtime"
)

$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $PackageRoot
$Venv = Join-Path $ProjectRoot ".venv"

if (-not (Test-Path $Venv)) {
    py -3.10 -m venv $Venv
}

$Python = Join-Path $Venv "Scripts\python.exe"
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
if ($Profile -eq "realtime") {
    & $Python -m pip install -r (Join-Path $ProjectRoot "requirements-windows.txt")
    if ($LASTEXITCODE -ne 0) { throw "CUDA/realtime dependency installation failed." }
    & $Python (Join-Path $PSScriptRoot "download_assets.py") --profile realtime-yolo
    & $Python (Join-Path $PSScriptRoot "doctor.py") --realtime
} else {
    & $Python -m pip install -r (Join-Path $ProjectRoot "requirements-cuda.txt") -r (Join-Path $ProjectRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "CUDA/offline dependency installation failed." }
    & $Python (Join-Path $PSScriptRoot "download_assets.py") --profile offline-demo
    & $Python (Join-Path $PSScriptRoot "doctor.py")
}

& $Python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda); assert torch.version.cuda is not None, 'CPU-only PyTorch detected'; print('CUDA available:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation verification failed." }

Write-Host "Environment ready. Start with:"
Write-Host "  & '$Python' -m mph_gait_id.ui_app"
