# AGENTS.md

## Build/Lint/Test Commands
- Dependencies: Install FLIR SDK from SDK installation/ directory
- Setup environment: `conda env create -f environment.yml` or `pip install -r requirements.txt`
- Lint: `python -m py_compile *.py` for syntax; optional `flake8`, `black --check`, `mypy *.py` for types
- Test: No unit tests; validate GUIs: `python SDK_dashboard.py`, `python hotspot_gui.py`; check outputs/errors
- Single script validation: Run `python <script>.py` and inspect for exceptions/output

## Code Style Guidelines
- **Imports**: Group stdlib, third-party, local; try/except for optional imports (PIL, cv2, fnv)
- **Formatting**: 4 spaces indent, ~100 chars/line max, blank lines for readability
- **Naming**: snake_case functions/variables, CamelCase classes, ALL_CAPS constants
- **Types**: Use type hints on functions/classes (e.g., `Optional[Tuple[int,int,int,int]]`)
- **Error Handling**: try/except with specific exceptions; use traceback.print_exc() or messagebox; no bare except
- **Docstrings**: """ for modules/classes/functions; describe purpose, params, returns
- **Comments**: # for inline explanations
- **Structure**: Private methods with _; constants at top; descriptive names; follow existing patterns</content>
<parameter name="filePath">AGENTS.md