$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $PackageRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run mph_gait_id\scripts\bootstrap_windows.ps1 first."
}

Push-Location $ProjectRoot
try {
    & $Python -m mph_gait_id.ui_app @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
