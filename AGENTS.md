# AGENTS.md

## Project Overview
- Firebrand Thermal Analysis dashboard for FLIR SDK SEQ files.
- Primary entry point: `FirebrandThermalAnalysis.py`.
- Packaged outputs are generated under `dist/` (ignored) and build artifacts under `build/`.
- Support modules: `SDK.py` and `tutorial/` examples.

## Setup
- Install FLIR SDK from `SDK/`.
- Create environment: `conda env create -f environment.yml`.
- Activate: `conda activate firebrand-thermal`.
- Install SDK wheel (per OS) from `SDK/`.
- Linux: if using the SDK installer, build a wheel from the installed SDK Python dir.

## Update Environment
- `conda env update -f environment.yml --prune`

## Run
- Dashboard: `python FirebrandThermalAnalysis.py`
- Single script: `python <script>.py`

## Build (PyInstaller)
- macOS: `./build/build_macos.sh`
- Windows: `.\build\build_windows.ps1` (single-file `dist/FirebrandThermalAnalysis.exe`)
- Linux: `./build/build_linux.sh`
- GitHub Actions: `.github/workflows/build.yml` (runs on `workflow_dispatch` and `v*` tags)
- Optional env vars:
  - `FLIR_SDK_WHEEL` (preferred) or `FLIR_SDK_PYTHON_DIR` + `FLIR_SDK_SHADOW_DIR`
  - `FLIR_SDK_LIB_DIR` + `FLIR_SDK_BIN_DIR` for SDK runtime libraries
- Build scripts include `--collect-all fnv` to bundle the SDK Python package.

## Package Installers
- macOS: `./build/package_macos_dmg.sh`
- Windows: `.\build\package_windows.ps1` (requires Inno Setup; wraps the single EXE)
- Linux: `./build/package_linux_appimage.sh` (requires `appimagetool`; includes a default icon)

## Lint/Test
- Syntax: `python -m py_compile *.py`
- Optional: `flake8`, `black --check`, `mypy *.py`
- No unit tests; validate GUIs by running them and checking for errors.

## Behavioral Expectations
- Frame numbers in UI/CSV are 1-based.
- Export end accepts `max` and defaults to full length per file.
- Emissivity shows metadata for the active SEQ; default input is 0.9.
- CSV export saves next to the SEQ with the same base name.
- Export CSV (all files) runs in parallel across files.
- Status text is prefixed with `Status:` for quick scanning.

## Code Style Guidelines
- **Imports**: Group stdlib, third-party, local; try/except for optional imports (PIL, cv2, fnv).
- **Formatting**: 4 spaces indent, ~100 chars/line max, blank lines for readability.
- **Naming**: snake_case functions/variables, CamelCase classes, ALL_CAPS constants.
- **Types**: Use type hints on functions/classes (e.g., `Optional[Tuple[int,int,int,int]]`).
- **Error Handling**: try/except with specific exceptions; use traceback.print_exc() or messagebox; no bare except.
- **Docstrings**: Use `"""` for modules/classes/functions; describe purpose, params, returns.
- **Comments**: `#` for inline explanations.
- **Structure**: Private methods with `_`; constants at top; descriptive names; follow existing patterns.
