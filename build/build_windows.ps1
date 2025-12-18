$ErrorActionPreference = "Stop"

# Build on Windows only. Requires FLIR SDK and PyInstaller installed.
# Optional env vars:
# - PYTHON_BIN: python executable (default: python)
# - FLIR_SDK_LIB_DIR: directory containing FLIR SDK DLLs
# - FLIR_SDK_BIN_DIR: directory containing FLIR SDK binaries

$AppName = "FirebrandThermalAnalysis"
$Entry = "SDK_dashboard.py"
$Python = $env:PYTHON_BIN
if (-not $Python) { $Python = "python" }

$opts = @("--windowed","--onedir","--name",$AppName,$Entry)

if ($env:FLIR_SDK_LIB_DIR) { $opts += @("--add-binary","$env:FLIR_SDK_LIB_DIR\\*;.") }
if ($env:FLIR_SDK_BIN_DIR) { $opts += @("--add-binary","$env:FLIR_SDK_BIN_DIR\\*;.") }

& $Python -m PyInstaller @opts
Write-Host "Build output: dist\$AppName"
