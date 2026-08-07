$ErrorActionPreference = "Stop"

# Build on Windows only. Uses uv to provision the build env.
# The project's pyproject.toml pins the per-OS FLIR SDK wheel via [tool.uv.sources],
# so `uv sync` handles all dependencies (including the dev group: pyinstaller).
# Optional env vars (custom SDK overrides):
# - FLIR_SDK_WHEEL: path to a prebuilt FileSDK wheel (preferred)
# - FLIR_SDK_PYTHON_DIR: FLIR SDK python folder containing setup.py
# - FLIR_SDK_SHADOW_DIR: shadow dir for wheel build (default: C:\temp\flir_sdk_build)
# - FLIR_SDK_LIB_DIR: directory containing FLIR SDK DLLs
# - FLIR_SDK_BIN_DIR: directory containing FLIR SDK binaries

$AppName = "FirebrandThermalAnalysis"
$Entry = "FirebrandThermalAnalysis.py"
$IconPath = "docs\\logo.ico"
$LogoData = "docs\\branding\\logo-square.png;docs\\branding"

# Write VERSION file from git tag (CI sets APP_VERSION env var, else use git)
$Version = $env:APP_VERSION
if (-not $Version) {
  $Version = (git describe --tags --abbrev=0 2>$null)
}
if (-not $Version) { $Version = "dev" }
Set-Content -Path "VERSION" -Value $Version -NoNewline
Write-Host "Building version: $Version"

uv sync --group dev

if ($env:FLIR_SDK_WHEEL) {
  uv pip install --reinstall-package filesdk $env:FLIR_SDK_WHEEL
} elseif ($env:FLIR_SDK_PYTHON_DIR) {
  $ShadowDir = $env:FLIR_SDK_SHADOW_DIR
  if (-not $ShadowDir) { $ShadowDir = "C:\\temp\\flir_sdk_build" }
  New-Item -ItemType Directory -Force -Path $ShadowDir | Out-Null
  uv run python "$env:FLIR_SDK_PYTHON_DIR\\setup.py" bdist_wheel --shadow-dir $ShadowDir
  $Wheel = Get-ChildItem -Path "$ShadowDir\\dist" -Filter *.whl | Select-Object -First 1
  if ($Wheel) {
    uv pip install --reinstall-package filesdk $Wheel.FullName
  } else {
    Write-Error "No wheel found in $ShadowDir\\dist"
    exit 1
  }
}

$opts = @(
  "--windowed",
  "--onefile",
  "--noconfirm",
  "--name", $AppName,
  "--icon", $IconPath,
  "--add-data", $LogoData,
  "--add-data", "VERSION;.",
  "--collect-all", "fnv",
  "--collect-all", "tkinterdnd2",
  $Entry
)

if ($env:FLIR_SDK_LIB_DIR) { $opts += @("--add-binary","$env:FLIR_SDK_LIB_DIR\\*;.") }
if ($env:FLIR_SDK_BIN_DIR) { $opts += @("--add-binary","$env:FLIR_SDK_BIN_DIR\\*;.") }

uv run --group dev python -m PyInstaller @opts
Write-Host "Build output: dist\\$AppName.exe"