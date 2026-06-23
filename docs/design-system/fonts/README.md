# Fonts

This system uses three families, all available on Google Fonts:

- **Fira Sans** (UI) — already in the source codebase. Weights 400/500/600/700.
- **Fira Code** (data / monospace) — already in the source codebase. Weights 400/500/600.
- **IBM Plex Sans** (display, marketing only) — weights 500/600.

Currently loaded via Google Fonts CDN in `colors_and_type.css`. To host locally:

1. Download `.ttf` / `.woff2` files from https://fonts.google.com/
2. Drop them in this directory.
3. Replace the `@import url(...)` block in `colors_and_type.css` with `@font-face` declarations pointing at the local files.

## Substitutions

None — every family in the system is open-source and available on Google Fonts.
The source codebase already references Fira Sans + Fira Code, so this is a 1:1 match.

## Pairing notes

- Fira Sans for everything UI: labels, buttons, section headers.
- Fira Code with **tabular numbers** (`font-variant-numeric: tabular-nums`) and **ligatures disabled** (`font-variant-ligatures: none`) for every numeric/data field. Disabling ligatures is important because `>=` should *not* render as `≥` in a CSV preview or threshold input.
- IBM Plex Sans only in marketing/about contexts — never in the live app chrome.
