$ErrorActionPreference = "Stop"

# Build on Windows only. Requires FLIR SDK and PyInstaller installed.
# Optional env vars:
# - PYTHON_BIN: python executable (default: python)
# - FLIR_SDK_LIB_DIR: directory containing FLIR SDK DLLs
# - FLIR_SDK_BIN_DIR: directory containing FLIR SDK binaries
# - FLIR_SDK_WHEEL: path to a prebuilt FileSDK wheel (preferred)
# - FLIR_SDK_PYTHON_DIR: FLIR SDK python folder containing setup.py
# - FLIR_SDK_SHADOW_DIR: shadow dir for wheel build (default: C:\temp\flir_sdk_build)

$AppName = "FirebrandThermalAnalysis"
$Entry = "FirebrandThermalAnalysis.py"
$Python = $env:PYTHON_BIN
if (-not $Python) { $Python = "python" }

$opts = @("--windowed","--onefile","--noconfirm","--name",$AppName,"--collect-all","fnv",$Entry)

if ($env:FLIR_SDK_WHEEL) {
  & $Python -m pip install $env:FLIR_SDK_WHEEL
} elseif ($env:FLIR_SDK_PYTHON_DIR) {
  $ShadowDir = $env:FLIR_SDK_SHADOW_DIR
  if (-not $ShadowDir) { $ShadowDir = "C:\\temp\\flir_sdk_build" }
  New-Item -ItemType Directory -Force -Path $ShadowDir | Out-Null
  & $Python "$env:FLIR_SDK_PYTHON_DIR\\setup.py" bdist_wheel --shadow-dir $ShadowDir
  $Wheel = Get-ChildItem -Path "$ShadowDir\\dist" -Filter *.whl | Select-Object -First 1
  if ($Wheel) {
    & $Python -m pip install $Wheel.FullName
  } else {
    Write-Error "No wheel found in $ShadowDir\\dist"
    exit 1
  }
}

if ($env:FLIR_SDK_LIB_DIR) { $opts += @("--add-binary","$env:FLIR_SDK_LIB_DIR\\*;.") }
if ($env:FLIR_SDK_BIN_DIR) { $opts += @("--add-binary","$env:FLIR_SDK_BIN_DIR\\*;.") }

& $Python -m PyInstaller @opts
Write-Host "Build output: dist\$AppName.exe"
