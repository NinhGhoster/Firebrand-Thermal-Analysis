---
name: firebrand-thermal-analysis-design
description: Use this skill to generate well-branded interfaces, mockups, and assets for Firebrand Thermal Analysis (FTA) — a desktop scientific tool for FLIR radiometric file analysis, firebrand detection, and thermal video playback. Contains the full design language: warm-dark surfaces, the ember-orange accent sampled from the Inferno colormap, Fira Sans / Fira Code typography, the side-panel pod system, the bottom-pinned control pod, and a high-fidelity React recreation of the redesigned dashboard.
user-invocable: true
---

Read the `README.md` file within this skill first, then explore the other available files.

## What's here

```
README.md                # full product context, content fundamentals, visual foundations, iconography
colors_and_type.css      # CSS variables — colors, type, spacing, motion, radii, surfaces
fonts/                   # Fira Sans, Fira Code, IBM Plex Sans (or Google Fonts notes)
assets/                  # logos (current + flagged), original screenshot, cropped thermal frame
preview/                 # design-system preview cards
ui_kits/dashboard/       # high-fidelity React recreation of the redesigned dashboard
  index.html             # interactive prototype, 1280×800
  *.jsx                  # reusable components (SidePanel, Canvas, ColorBar, ControlPod, ...)
  dashboard.css          # kit-local component styles
```

## When invoked

If creating visual artifacts (slides, mocks, marketing pages, throwaway prototypes), copy the assets and tokens you need into a new static HTML file and ship that file. Always:

1. Include `colors_and_type.css` (or inline the CSS variables) and use the semantic tokens — `--bg-app`, `--bg-card`, `--ember-500`, `--status-warn` — not raw hex values.
2. Use **Fira Sans** for all UI text and **Fira Code** for any numeric / status / data field. Disable Fira Code ligatures (`font-variant-ligatures: none`) and enable tabular numerals.
3. Keep the warm-dark palette: cards on `#1C2026`, hairline borders on `#262B33`, no drop shadows.
4. Use the **ember accent (`#FF6B1A`)** for primary actions and section headers — never generic blue.
5. Stick to the lab-instrument voice: no emoji, no marketing register, no first/second person, Title Case headers and sentence-case buttons. Numeric values are right-aligned and use the data-status amber (`#F5B83D`) for telemetry strings.
6. The Inferno colormap is the brand visual; use it for hero gradients, color bars, mark-and-logo accents.

If working on production code (the actual CustomTkinter app, a web rewrite, or marketing site), copy assets and read `README.md` to internalise the rules — then design as the expert.

If the user invokes this skill with no specific guidance, ask what they want to build (rebuild of the dashboard, slide deck, marketing site, etc.), ask 3–5 targeted questions, and then deliver high-fidelity HTML artifacts or production-shaped React code as needed.

## Things that are explicitly off-brand

- Bluish accents — replaced with ember.
- Drop shadows on chrome — only hairlines.
- Emoji or unicode icons (`▶`, `■`, `▲`).
- Rounded-corner cards with a coloured left-border stripe (saas tropes).
- "Loading…" or any conversational copy.
- AI-generated logos — the current `logo-banner.png` / `logo-square.png` are placeholders; replace them when commissioning a real mark.

## Source references

- Application repository: https://github.com/NinhGhoster/Firebrand-Thermal-Analysis
- Companion compressor: https://github.com/NinhGhoster/SEQ-CSQ-compressor
- FLIR Science File SDK: https://flir.custhelp.com/app/answers/detail/a_id/3504
