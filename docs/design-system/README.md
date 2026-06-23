# Firebrand Thermal Analysis — Design System

> A redesign-ready design system for the **Firebrand Thermal Analysis** (FTA) dashboard — a high-performance, desktop scientific tool for extracting and tracking thermal data from FLIR radiometric files in combustion experiments.

This system was reverse-engineered from the existing CustomTkinter application and refined into a cleaner, more confident research instrument. Use it to rebuild the app, prototype new screens, or produce marketing/research materials that share its visual DNA.

---

## Product Context

**What it is.** Firebrand Thermal Analysis is a GUI-driven dashboard used by wildfire and combustion researchers to load massive FLIR radiometric files (SEQ, CSQ, JPG, ATS, SFMOV, IMG, NC) and automatically detect, track, and export the temperature, area, and bounding-box statistics of *firebrands* (airborne embers) and *fuel-bed hotspots* during experiments.

**Who uses it.** Fire-science researchers — typical user is a PhD/postdoc analysing 30+ GB thermal video recordings from controlled burns. Cited project team: H. Nguyen, J. Filippi, T. Penman, M. Peace, A. Filkov.

**Where it lives.** Cross-platform desktop app (macOS, Windows, Linux). Built with Python + CustomTkinter + OpenCV + the Teledyne FLIR Science File SDK. Single window, no web component.

**Companion tool.** `SEQ-CSQ-compressor` — compresses raw FLIR videos to NetCDF4 (`.nc`) for ~70% size reduction while preserving 0.01 °C accuracy. FTA reads these `.nc` files natively.

**Core surfaces of the app:**
1. **Main canvas** — full-bleed thermal video playback with colormap applied, ROI overlay, tracking markers, hover temperature readout.
2. **Color bar** — vertical temperature-to-color gradient strip pinned to the right of the canvas with `min` / `max` °C labels.
3. **Side telemetry panel** — bento-stacked cards for Data Source, Visualisation, Parameters (threshold, emissivity, frame range, ROI), Exports, and footer credits.
4. **Bottom control pod** — full-width scrub slider, transport controls (Play / < / >), and status line.

## Source Materials

This system was built from the public source for the existing application. The reader should consult these to build deeper context:

- **Application & code** — [github.com/NinhGhoster/Firebrand-Thermal-Analysis](https://github.com/NinhGhoster/Firebrand-Thermal-Analysis)
  - `FirebrandThermalAnalysis.py` — the entire CustomTkinter dashboard (~76 KB single file). Source of truth for the UI structure, colours, and behaviour.
  - `AGENTS.md` — coding conventions, keyboard shortcuts, behavioural expectations.
  - `README.md` — user-facing feature list and CSV schema.
  - `docs/branding/` — current AI-generated logo (banner + square).
  - `docs/screenshots/interface-overview-2026-04-14.png` — current UI screenshot.
- **Companion compressor** — [github.com/NinhGhoster/SEQ-CSQ-compressor](https://github.com/NinhGhoster/SEQ-CSQ-compressor)
- **FLIR Science File SDK (Python)** — [flir.custhelp.com/app/answers/detail/a_id/3504](https://flir.custhelp.com/app/answers/detail/a_id/3504)

Nothing in this design system is auto-generated from those repos — read what you need from them directly to refine further.

---

## Index — What's In This Folder

```
README.md                   # this file
SKILL.md                    # Agent-Skill manifest (cross-compat with Claude Code)
colors_and_type.css         # CSS variables for colours, typography, spacing
fonts/                      # local webfonts (or notes on substitutes)
assets/                     # logos, screenshots, raw imagery from the app
preview/                    # design-system preview cards (rendered in the DS tab)
ui_kits/
  dashboard/                # high-fidelity React recreation of the FTA dashboard
    README.md
    index.html              # interactive click-through of the redesigned dashboard
    *.jsx                   # modular components (SidePanel, Canvas, ControlPod, ...)
```

---

## CONTENT FUNDAMENTALS

The application's copy is **technical, terse, and precise** — written for a researcher who knows what an "emissivity" or "ROI" is and doesn't want their hand held. Buttons are verbs or short noun-phrases; labels never use full sentences; status text is prefixed for scanability.

**Voice & tone.**
- **Lab-instrument register.** Like a high-end oscilloscope, not a consumer app. No marketing voice. No exclamation points. No emoji. (Confirmed by full codebase grep — zero emoji used in UI strings.)
- **No first / second person.** The interface never says "you" or "we". It describes the system state or the action. Example: not *"Click here to open your file"* but **"Open File/Folder"** on the button itself.
- **Declarative, not conversational.** Status line reads `Status: frame 8854/17318 | ROI: (3, 7, 1020, 373) | thresh: 300.0C` — a key/value telemetry stream, not a sentence.
- **Imperative for actions.** `Apply`, `Export`, `Reset ROI`, `Auto Target Fuel`, `Set Start`. Every button is a single verb or a verb + object.

**Casing.**
- **Title Case** for section headers in the side panel: `Data Source`, `Visualisation`, `Parameters`. (Note British English `Visualisation` — preserve it.)
- **Sentence case** for button labels: `Open File/Folder`, `Reset Zoom`, `Auto Target Fuel`.
- **lowercase** for inline data and status fragments: `frame 8854/17318`, `max`, `counts`, `not set`, `ready`.
- **ALL CAPS** is **never** used. No screaming. No marketing badges.

**Numbers & units.**
- Frame numbers are 1-based, always shown as `N` or `N/total` (e.g. `8854/17318`).
- Temperatures show 1 decimal with the °C symbol or `C` suffix: `306.9C`, `300.0`.
- Emissivity shows 3 decimals: `0.940`.
- ROI is always the 4-tuple `(x, y, w, h)` — comma-separated, no labels needed.
- Percentages have no decimal: `100%`, never `100.0%`.
- `max` is a reserved keyword in the End-frame field meaning "use full file length".

**Status line conventions.**
- Always prefixed with `Status:` — verified in source: `self._base_status = "Status: ready"`.
- Pipe-separated key/value pairs after the prefix.
- Lowercase keys, raw values: `Status: opened Rec-0158.seq | 1024x768 | 17318 frames`.
- Errors surface in modal dialogs (`messagebox.showerror`), not the status line.

**Microcopy examples (verbatim from source).**
| Surface | String |
|---|---|
| Open menu | `Open file(s)...` · `Open folder...` |
| Empty state | `Auto ROI: (not set)` |
| Default colormap | `Inferno` |
| Set-range hover | `Start = 8854` · `End = 8854` |
| Footer credit | `Developed by H. Nguyen, J. Filippi, T. Penman, M. Peace, A. Filkov (v0.0.3)` |
| Update prompt | `New version v0.0.4 is available (current v0.0.3). Open the release page?` |

**What to avoid.**
- Don't pad UI with friendly explanations. If a control needs explanation, its label is wrong.
- Don't say *"Loading…"* — say *"Status: opening Rec-0158.seq"*.
- Don't invent new tone (no jokes, no encouragement, no "Great!").
- Don't use emoji or unicode decorations (`✓`, `→`, `⚡` are all off-brand).
- Don't translate British English to American — keep `Visualisation`, `colour bar`, `analyse` if they're already there.

---

## VISUAL FOUNDATIONS

The redesign is a **scientific instrument**, not a SaaS dashboard. Heavy use of dark surfaces, a single warm accent borrowed from the thermal data itself (Inferno colormap), monospace numerals everywhere data is shown, and a strict bento grid.

### Colour

**Surfaces — warm-dark neutrals.** The original app uses Tailwind `slate-900` / `slate-800` (cool blue-grey). The redesign shifts to a **slightly warmer near-black** that sits better next to the orange/red thermal imagery — `#0B0D10` background, `#14171C` raised surface, `#1C2026` panel, `#262B33` hairline borders. The cooler slate looked clinical *and* fought the inferno palette; the warm-dark feels like a darkroom around a hot subject.

**Accent — Ember Orange.** Replace the generic blue (`#3B82F6`) with a thermal-derived primary: `#FF6B1A` (Ember 500) for primary actions, with `#FFB347` (Ember 300) for hovers/highlights. This is sampled directly from the Inferno colormap that defines the product. The blue felt arbitrary; the orange ties the chrome to the data.

**Telemetry yellow.** Status text keeps its amber tone (`#F5B83D`) — high-contrast on dark, scannable from across a lab.

**Semantic colours.** Pulled from the same Inferno gradient so warnings/successes don't feel imported from a different system:
- Success / OK detection: `#5BD08A` (cool but desaturated, only used for confirmations)
- Warning: `#F5B83D` (telemetry amber)
- Critical: `#E0245E` (deep magenta-red, from the Inferno low-mid range)
- Info: `#7AB8FF` (used rarely; only for "update available" type chrome)

**Thermal colourmap palette.** Inferno is the brand colormap — Black → Deep Purple → Magenta → Orange → Yellow → White. All eight maps in `COLORMAPS` (Inferno, Jet, Hot, Magma, Plasma, Bone, Turbo, Grayscale) are first-class product data and are previewed in the DS tab.

### Typography

- **UI sans:** the app uses `Fira Sans`. The redesign keeps Fira Sans for labels and buttons — it's a clean grotesque with strong Unicode coverage including math symbols, and it's already in the codebase. Headers use the same family with weight 600.
- **Numeric / mono:** `Fira Code` already powers the data fields. **Keep it.** Numeric tables, status text, ROI tuples, frame counters, and the °C readout all use Fira Code with tabular figures so digits don't jitter when the count ticks. Ligatures are **disabled** in the data context (`font-variant-ligatures: none`) because `>=` shouldn't render as `≥` in a CSV preview.
- **Display:** an optional `IBM Plex Sans` (semi-bold) is used in marketing/about surfaces only — not in the live app chrome.

If any font is missing locally we fall back to Google Fonts (Fira Sans, Fira Code, IBM Plex Sans all available). Substitutions are flagged in `fonts/README.md`.

### Spacing & Layout

- Base unit `4px`. Scale: `4 / 8 / 12 / 16 / 20 / 24 / 32 / 40 / 56 / 72`.
- **Bento grid.** The current app already uses a bento layout (canvas + side panel + bottom pod). Redesign keeps it, with tighter gaps (`10px → 8px` between panels) and a consistent `12px` inner padding on every card.
- **Side panel width** locked at `340px` (current value, retained).
- **Canvas** is the only fluid region — it consumes everything else.
- **Color bar** is `36px` wide (up from `30px`) with a `12px` gap from the canvas for visual breathing room.

### Backgrounds, surfaces, depth

- **No gradients on chrome.** Surfaces are flat solid fills. Gradients are reserved for the **thermal data** itself (colormap strip, hero imagery).
- **Hairlines, not shadows.** Cards have a 1px inner border `#262B33` and no drop shadow. The original app uses `corner_radius=10` on CTkFrame and no shadow — preserve that. Outer drop shadows are explicitly off-brand: they read as "web app", not "instrument".
- **Subtle grid overlay** on the marketing/about background only — 32px square grid in `#262B33` at 30% opacity. Never on the live app canvas.
- **Imagery is warm and high-contrast** — favour the inferno-palette thermal frames over photographic source frames in marketing. When photo imagery is used, treat it warm (slight orange grade, never cool).

### Borders & radii

- **Corner radius scale**: `0 / 4 / 6 / 10 / 16`. Pods use `10` (matches `corner_radius=10` in source). Buttons use `6`. Inputs use `4`. Avatars / chips can use `16` (pill).
- **Border thickness**: always 1px. Never 2px+ except for the green ROI rectangle on canvas (kept `#10B981`, 2px outlined — instantly recognisable from the original).
- **Focus ring**: 2px ember outline with 2px offset on keyboard focus, never on mouse hover.

### Hover / press / active states

- **Hover (buttons)**: background lifts one step in the neutral scale (`#1C2026` → `#262B33`), or for primary actions, the ember saturates by ~8% (`#FF6B1A` → `#FF7E35`). No scale or translate on hover.
- **Press**: background darkens one step from rest; no shrink/scale animation. A 1px inset top shadow `rgba(0,0,0,0.25)` simulates the press without being cartoonish.
- **Active / selected** (e.g. current colormap): 2px left ember stripe inside the row + `#1C2026` background. No outlined "selected" pill.
- **Disabled**: 40% opacity on text/fill, cursor `not-allowed`, no interaction. Maintains layout.

### Animation

- **Sparse.** This is an instrument; nothing wiggles. Default policy: only animate state changes that aid comprehension.
- **Easing**: a single curve — `cubic-bezier(0.32, 0.72, 0, 1)` (sharp out, slow finish — feels machined, not springy). Token: `--ease-instrument`.
- **Duration**: `120ms` for micro (hover, focus ring), `200ms` for panels and tooltips, `320ms` for layout shifts. Anything longer is wrong.
- **No bounces, no springs, no slide-in entrances, no skeleton shimmer.** The thermal canvas is the only thing that moves at frame rate.
- **Status text changes** crossfade for `120ms` to avoid jitter when telemetry updates 30×/s — but if reduced-motion is on, swap immediately.

### Transparency & blur

- **Transparency** is used sparingly: 70% black scrim on tooltips over the canvas; 60% on toast notifications. Surfaces themselves are opaque.
- **Backdrop blur** only on the canvas tooltip (`backdrop-filter: blur(8px)`) so temperature readouts stay legible over hot pixels.

### Cards (chrome / panel components)

The standard side-panel pod:
- Background: `#1C2026`
- Border: `1px solid #262B33`
- Radius: `10px`
- Inner padding: `12px`
- Header: Fira Sans 14/600 in **ember accent** (`#FF6B1A`) — replaces the old blue header. Section heading lives on its own line above the controls.
- No shadow, no gradient, no decoration.

### Inputs & buttons

- **Text input**: `32px` tall, `#0B0D10` fill (inset, slightly darker than surrounding), 1px `#262B33` border, 4px radius, Fira Code text, 12px horizontal padding.
- **Primary button**: ember fill, white text, Fira Sans 13/500, `32px` tall, 6px radius. The standard CTA across the app.
- **Secondary button**: `#262B33` fill, near-white text. For navigation, paired actions.
- **Icon button**: 32×32, no fill at rest, `#262B33` on hover, ember icon.
- **Combobox** (visualisation dropdown): matches text input chrome; chevron is a 12px ember glyph.

### Layout rules / fixed elements

- The bottom control pod is **always full-width and pinned**. Never floats or scrolls away.
- The side panel scrolls internally when content overflows; the main canvas never scrolls.
- The colour bar is always visible when a file is loaded; hidden in the empty state.
- The slider is row-1 in the pod (full bleed); the transport buttons + status are row-2. Never re-order these.

### Imagery treatment

- **Thermal frames** (inferno-palette) are the hero visual. When used in marketing, leave them at native saturation — they *are* the brand.
- **Photographic frames** (visible-spectrum reference) get a slight warm grade (`+5` warmth, `−5` saturation) so they don't fight the chrome.
- Avoid stock photography entirely. The product's own data is the imagery system.

---

## ICONOGRAPHY

The existing app **uses no iconography**. Every control is a text label — `Open File/Folder`, `Reset Zoom`, `<`, `>`, `<<`, `>>`, `Play`, `Pause`. This is intentional and on-brand for an instrument; it should mostly be preserved.

**The redesign introduces icons only where a label is impractical:**
- Transport controls (Play / Pause / Step Forward / Step Back / Stop)
- Toolbar icons in the colourmap & zoom row
- Status-line indicators (recording, exporting, error)

**Icon set:** **Lucide** (CDN: `https://unpkg.com/lucide-static@latest/icons/`). Chosen because:
- Stroke-based, 1.5px stroke, 24px box — matches the instrument aesthetic.
- Permissive ISC licence.
- Has every glyph we need: `play`, `pause`, `chevron-left/right`, `chevrons-left/right`, `square` (stop), `zoom-in/out`, `maximize-2`, `folder-open`, `download`, `settings-2`, `crosshair` (for ROI), `thermometer`, `flame`.
- **Substitute flag:** the original codebase has **no built-in icon system**, so this is a substitution — the redesign introduces icons that were never in the source. Flagged so the user can approve the addition.

**Usage rules:**
- Icons render at **16px** in compact rows, **20px** in transport controls.
- Stroke colour: inherit `currentColor`. Default to text colour, not the ember accent — except active/pressed states.
- **Never use filled (solid) variants.** Stroke only — consistent with the instrument feel.
- **Never use emoji.** Verified absent from current code; keep it that way.
- **Never use unicode symbols as icons.** No `▶`, `■`, `◀`. Use Lucide SVGs or text labels.

**Logos & marks:**
- Current logos in `docs/branding/` are AI-generated (visible DALL·E artifacts: rainbow gradient streak across the flame, awkward `FTA` letterforms inside a circle). They are usable as placeholders but **flagged for replacement** with a properly designed mark.
- The system ships a clean **wordmark** built from Fira Sans 700 + a small flame glyph (Lucide `flame`) — see `preview/logo-wordmark.html`.
- A square avatar variant ships in `assets/logo-mark.svg` (placeholder until a real mark is commissioned).

---

## Caveats & Open Questions

- **Logos are AI-generated and should be replaced.** Treated as placeholders.
- **Inter, Roboto, system stacks avoided.** Kept the codebase's Fira Sans / Fira Code combo.
- **Icon system is an addition, not a port.** Source app has no icons; Lucide is suggested as a future direction.
- **British English preserved** (`Visualisation`, `colour`) — confirm with project team before normalising.
- **No real video frames bundled.** The single screenshot at `assets/interface-overview-2026-04-14.png` is the only on-product imagery available — when running the rebuilt UI we mock the canvas with a placeholder thermal image.

---

See `SKILL.md` for the agent-skill manifest and `ui_kits/dashboard/` for the high-fidelity React recreation of the redesigned dashboard.
