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
if ($Profile -eq "realtime") {
    & $Python -m pip install -r (Join-Path $ProjectRoot "requirements-windows.txt")
    & $Python (Join-Path $PSScriptRoot "download_assets.py") --profile realtime-yolo
    & $Python (Join-Path $PSScriptRoot "doctor.py") --realtime
} else {
    & $Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
    & $Python (Join-Path $PSScriptRoot "download_assets.py") --profile offline-demo
    & $Python (Join-Path $PSScriptRoot "doctor.py")
}

Write-Host "Environment ready. Start with:"
Write-Host "  & '$Python' -m mph_gait_id.ui_app"
