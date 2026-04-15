# AGENTS.md

## Project Overview
- Firebrand Thermal Analysis dashboard for FLIR SDK radiometric files (SEQ, CSQ, JPG, ATS, SFMOV, IMG).
- Primary entry point: `FirebrandThermalAnalysis.py`.
- UI/UX based on **CustomTkinter** featuring a Bento Grid layout and deep dark mode.
- Packaged outputs are generated under `dist/` (ignored) and build artifacts under `build/`.
- Support modules: `SDK.py` and `tutorial/` examples.
- Thermal colormaps: Inferno (default), Jet, Hot, Magma, Plasma, Bone, Turbo, Grayscale.
- Canvas supports zoom (scroll / +/-), pan (middle-drag), and hover temperature readout.
- Color bar displays temperature-to-color gradient alongside the canvas.

## Setup
- Install FLIR SDK from `SDK/`.
- Create environment: `conda env create -f environment.yml` (includes `customtkinter`).
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
- Linux: `./build/package_linux_appimage.sh` (requires `appimagetool`; uses `docs/branding/logo-square.png`)

## Release Lessons
- Do not assume "works in conda" means "works in packaged app". Source runs can see the repo tree; PyInstaller builds cannot unless files are explicitly bundled.
- `v0.0.3` regressed compared with `v0.0.2` because the app was changed to load branding assets from `docs/branding/logo-square.png`, but the build scripts were still using direct PyInstaller CLI builds that did not bundle that asset.
- The packaged macOS app also failed at startup with `customtkinter not found in libs/` because the code still assumed a source-tree `libs/` folder. Packaged apps must import from bundled modules first and only fall back to local `libs/` during source runs.
- **Critical Fix Applied:** To fix this, always guard path injections with `if not getattr(sys, "frozen", False):`. Furthermore, PyInstaller must explicitly be told to crawl local dependency folders during the build phase; you MUST include `--paths libs` in all three `build_*.sh/.ps1` scripts!
- The root rule: whenever a runtime path changes, update both the application code and every platform build script (`build_macos.sh`, `build_windows.ps1`, `build_linux.sh`, packaging scripts if relevant).
- The build scripts currently do not use `FirebrandThermalAnalysis.spec`; they invoke PyInstaller directly. Any icon/data/bundle-identifier change must therefore be reflected in the scripts too, not just in the `.spec` file.
- For frozen apps, use the `_MEIPASS`-aware resource helper pattern rather than `os.path.dirname(__file__)` alone.
- If the macOS release downloads but does not open, check Gatekeeper before assuming the app is broken:
  - `spctl -a -vv "/Applications/Firebrand Thermal Analysis.app"`
  - `xattr -l "/Applications/Firebrand Thermal Analysis.app"`
- A `rejected` result with `com.apple.quarantine` means macOS is blocking an unsigned/unnotarized download. That is separate from app crashes.
- Current macOS GitHub releases are not notarized. Users may need to remove quarantine manually unless proper Apple signing/notarization is added to CI.
- Tag-triggered GitHub releases only happen on `v*` refs. `workflow_dispatch` on `main` builds artifacts but skips the `release` job by design.

## Packaging Checklist
- Before shipping, launch the app from source and from a packaged artifact.
- On macOS, test the real bundle binary directly:
  - `"/Applications/Firebrand Thermal Analysis.app/Contents/MacOS/FirebrandThermalAnalysis"`
- Verify branding assets are bundled and used on all platforms:
  - `docs/logo.icns`
  - `docs/logo.ico`
  - `docs/branding/logo-square.png`
- If updating icons or logos, regenerate platform icon files and verify the build scripts still reference the current paths.
- If reusing an existing release tag, remember that it requires moving the tag and force-pushing it; otherwise the `release` job will publish the old artifact set.

## Lint/Test
- Syntax: `python -m py_compile *.py`
- Tests: `python -m pytest tests/ -v`
- Optional: `flake8`, `black --check`, `mypy *.py`

## Keyboard Shortcuts
| Key | Action |
|---|---|
| `Space` | Play / Pause |
| `S` | Stop |
| `Left` / `,` | Previous frame |
| `Right` / `.` | Next frame |
| `Home` | Jump to first frame |
| `End` | Jump to last frame |
| `+` / `-` | Zoom in / out |
| `0` | Reset zoom |
| `1`–`8` | Quick-select colormap |
| `R` | Reset ROI |
| `F` | Toggle fullscreen |
| `Escape` | Exit fullscreen |
| `Double-click` | Reset zoom |
| `Scroll wheel` | Zoom on cursor |
| `Middle-drag` | Pan view |

## Behavioral Expectations
- Frame numbers in UI/CSV are 1-based.
- Export end accepts `max` and defaults to full length per file.
- Emissivity shows metadata for the active SEQ; default input is 0.9.
- CSV export saves next to the SEQ with the same base name.
- Export CSV (all files) runs in parallel across files.
- Status text is prefixed with `Status:` for quick scanning.
- Color bar gradient updates when paused; skipped during playback for performance.
- Single-char shortcuts are suppressed when focus is in an Entry/Combobox.

## Code Style Guidelines
- **Imports**: Group stdlib, third-party, local; try/except for optional imports (PIL, cv2, fnv).
- **Formatting**: 4 spaces indent, ~100 chars/line max, blank lines for readability.
- **Naming**: snake_case functions/variables, CamelCase classes, ALL_CAPS constants.
- **Types**: Use type hints on functions/classes (e.g., `Optional[Tuple[int,int,int,int]]`).
- **Error Handling**: try/except with specific exceptions; use traceback.print_exc() or messagebox; no bare except.
- **Docstrings**: Use `"""` for modules/classes/functions; describe purpose, params, returns.
- **Comments**: `#` for inline explanations.
- **Structure**: Private methods with `_`; constants at top; descriptive names; follow existing patterns.
