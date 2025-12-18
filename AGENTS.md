# AGENTS.md

## Project Overview
- Thermal firebrand detection and tracking dashboard for FLIR SDK SEQ files.
- Primary entry points: `SDK_dashboard.py`, `hotspot_gui.py`, `hotspot_detector.py`.

## Setup
- Install FLIR SDK from `SDK installation/`.
- Create environment: `conda env create -f environment.yml` or `pip install -r requirements.txt`.

## Run
- Dashboard: `python SDK_dashboard.py`
- Hotspot GUI: `python hotspot_gui.py`
- Single script: `python <script>.py`

## Lint/Test
- Syntax: `python -m py_compile *.py`
- Optional: `flake8`, `black --check`, `mypy *.py`
- No unit tests; validate GUIs by running them and checking for errors.

## Code Style Guidelines
- **Imports**: Group stdlib, third-party, local; try/except for optional imports (PIL, cv2, fnv).
- **Formatting**: 4 spaces indent, ~100 chars/line max, blank lines for readability.
- **Naming**: snake_case functions/variables, CamelCase classes, ALL_CAPS constants.
- **Types**: Use type hints on functions/classes (e.g., `Optional[Tuple[int,int,int,int]]`).
- **Error Handling**: try/except with specific exceptions; use traceback.print_exc() or messagebox; no bare except.
- **Docstrings**: Use `"""` for modules/classes/functions; describe purpose, params, returns.
- **Comments**: `#` for inline explanations.
- **Structure**: Private methods with `_`; constants at top; descriptive names; follow existing patterns.
