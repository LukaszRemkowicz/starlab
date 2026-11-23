# Build dist\StarEffectsLab-<ver>-Windows-x64-Setup.exe (x64, unsigned). Run from anywhere; needs uv, cmake (VS Build Tools),
# Inno Setup 6 (ISCC) and the Photoshop SDK headers.
# Usage: installer\build-windows.ps1 [-Sdk DIR] [-Iscc PATH]
param(
    [string]$Sdk = "",
    [string]$Iscc = "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$here = $PSScriptRoot
$repo = Split-Path $here -Parent
# Build outside the home folder: tools record build paths (script launchers, bytecode, compiled kernels).
$stage = Join-Path $env:PUBLIC "star-effects-lab-build"
$app = Join-Path $stage "app"
$py = Join-Path $app "python"
# The SDK headers are the same for both systems: use whichever copy is present.
if (-not $Sdk) { $Sdk = @("win", "mac") | ForEach-Object { Join-Path $repo "photoshop\sdk\$_" } | Where-Object { Test-Path $_ } | Select-Object -First 1 }
$ver = ((Select-String -Path (Join-Path $repo "pyproject.toml") -Pattern '^version\s*=\s*"(.*)"')[0]).Matches[0].Groups[1].Value
# cmake ships with VS Build Tools but is not on PATH outside a developer prompt.
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $cmake = if (Test-Path $vswhere) { & $vswhere -latest -products * -find "**\CMake\bin\cmake.exe" | Select-Object -First 1 }
    if (-not $cmake) { throw "cmake not found: install CMake or the C++ CMake tools of VS Build Tools" }
    $env:PATH = (Split-Path $cmake) + ";" + $env:PATH
}
# Windows PowerShell 5.1 turns native stderr into errors under "Stop" when output is redirected: relax it for native calls.
function Run {
    if (-not (Get-Command $args[0] -ErrorAction SilentlyContinue)) { throw "not found: $($args[0])" }
    $ErrorActionPreference = "Continue"
    & $args[0] $args[1..($args.Count - 1)]
    if ($LASTEXITCODE) { throw "failed: $args" }
}
function Capture {
    if (-not (Get-Command $args[0] -ErrorAction SilentlyContinue)) { throw "not found: $($args[0])" }
    $ErrorActionPreference = "Continue"
    $r = & $args[0] $args[1..($args.Count - 1)] 2>$null
    if ($LASTEXITCODE) { throw "failed: $args" }
    $r
}

Set-Location $repo
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $app | Out-Null

Write-Host "== wheel"
$src = Join-Path $stage "source"
New-Item -ItemType Directory -Force $src | Out-Null
Copy-Item pyproject.toml, hatch_build.py, LICENSE, src $src -Recurse
Get-ChildItem $src -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
$env:UV_CACHE_DIR = Join-Path $stage "uv-cache"   # kernels record their build env path
Run uv build --wheel --directory $src
Remove-Item Env:UV_CACHE_DIR
$wheel = (Get-ChildItem (Join-Path $src "dist") -Filter "star_effects_lab-*.whl")[0].FullName

Write-Host "== bundled Python"
Run uv python install 3.14
$pyhome = (Capture uv run --no-project --managed-python --python 3.14 python -I -c "import sys; print(sys.base_prefix)").Trim()
if (-not (Test-Path (Join-Path $pyhome "python314.dll"))) { throw "python314.dll missing in $pyhome" }
Copy-Item $pyhome $py -Recurse
Remove-Item (Join-Path $py "EXTERNALLY-MANAGED"), (Join-Path $py "Lib\EXTERNALLY-MANAGED") -ErrorAction SilentlyContinue
# The app only uses PySide6 essentials: skip addons/webengine/pdf. The export is the full locked closure,
# so --no-deps installs exactly the locked versions.
(Capture uv export --no-dev --no-hashes --no-emit-project) |
    Where-Object { $_ -notmatch '^pyside6[-_](addons|webengine|pdf)' } |
    Set-Content (Join-Path $stage "requirements.txt") -Encoding ascii
Run uv pip install --python (Join-Path $py "python.exe") --no-deps --break-system-packages `
    -r (Join-Path $stage "requirements.txt") $wheel
Get-ChildItem (Join-Path $py "Lib\site-packages\PySide6") -Include designer.exe, assistant.exe, linguist.exe -Recurse -ErrorAction SilentlyContinue | Remove-Item
Get-ChildItem $py -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
# Program Files is not writable for users: ship bytecode. Exit status ignored (stdlib has deliberately broken test files).
try { & cmd /c "`"$py\python.exe`" -I -m compileall -q -j 4 `"$py\Lib`" >nul 2>&1" } catch {}

Write-Host "== launcher"
$ico = Join-Path $repo "src\starlab\presentation\qt\assets\application.ico"
Run cmake -S (Join-Path $here "launcher") -B (Join-Path $stage "launcher") -A x64 "-DPYTHON_HOME=$py" "-DICON_FILE=$ico"
Run cmake --build (Join-Path $stage "launcher") --config Release
Copy-Item (Join-Path $stage "launcher\Release\Star Effects Lab.exe") $app

Write-Host "== plug-in"
Run powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo "photoshop\build.ps1") -Sdk $Sdk `
    -App "C:\Program Files\Star Effects Lab\Star Effects Lab.exe"
# build.ps1 only self-tests against a dev .venv; test against the bundled runtime instead (from $repo: see build.ps1).
Run (Join-Path $repo "photoshop\build\Release\bridge_selftest.exe") (Join-Path $py "python.exe")
$plugin = Join-Path $repo "photoshop\build\Release\StarEffectsLab.8bf"

Write-Host "== setup"
$out = Join-Path $repo "dist"
Run $Iscc "/DAppVersion=$ver" "/DStageDir=$app" "/DPluginFile=$plugin" "/DOutDir=$out" "/DIconFile=$ico" (Join-Path $here "windows\star-effects-lab.iss")
$setup = Join-Path $out "StarEffectsLab-$ver-Windows-x64-Setup.exe"
"`nBuilt: $setup ({0:N1} MB)" -f ((Get-Item $setup).Length / 1MB)
