# Build the Star Effects Lab Photoshop filter on Windows (x64, Visual Studio Build Tools) and check it
# against the Python app.
# Usage: photoshop\build.ps1 [-Sdk DIR] [-App LAUNCHER]
#   -Sdk  Adobe Photoshop SDK (the folder holding pluginsdk\, or pluginsdk\ itself). Default: photoshop\sdk\win
#   -App  star-effects-lab launcher the filter starts. Default: this repo's .venv\Scripts\star-effects-lab.exe
param(
    [string]$Sdk = "",
    [string]$App = ""
)
$ErrorActionPreference = "Stop"

$here = $PSScriptRoot
$repo = Split-Path $here -Parent
if (-not $Sdk) { $Sdk = Join-Path $here "sdk\win" }
if (-not $App) { $App = Join-Path $repo ".venv\Scripts\star-effects-lab.exe" }

if (Test-Path (Join-Path $Sdk "pluginsdk")) { $Sdk = Join-Path $Sdk "pluginsdk" }
if (-not (Test-Path (Join-Path $Sdk "photoshopapi\photoshop"))) {
    Write-Error ("Photoshop SDK not found at: $Sdk`n" +
        "Download the Photoshop Plug-In SDK from https://developer.adobe.com/console (free, Adobe login),`n" +
        "unzip it to photoshop\sdk\win (so photoshop\sdk\win\pluginsdk exists), or pass -Sdk DIR.")
}
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Error "Missing cmake (install it, or the 'C++ CMake tools' part of Visual Studio Build Tools)."
}
if (-not (Test-Path $App)) {
    Write-Warning "Launcher not found at $App; the filter will fail until it exists (uv sync)."
}

$build = Join-Path $here "build"
cmake -S (Join-Path $here "plugin") -B $build -A x64 "-DPHOTOSHOP_SDK=$Sdk" "-DSTARLAB_COMMAND=$App"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
cmake --build $build --config Release --parallel
if ($LASTEXITCODE) { exit $LASTEXITCODE }

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (Test-Path $python) {
    # Run from the repo: python -c puts the current folder first on sys.path, so stray .py files there would shadow modules.
    Push-Location $repo
    & (Join-Path $build "Release\bridge_selftest.exe") $python
    $code = $LASTEXITCODE
    Pop-Location
    if ($code) { exit $code }
} else {
    Write-Warning "Skipping bridge self-test: $python not found (run uv sync)."
}

Write-Host ""
Write-Host "Plug-in: $build\Release\StarEffectsLab.8bf"
Write-Host "Install: copy it into C:\Program Files\Common Files\Adobe\Plug-Ins\CC\ and restart Photoshop."
