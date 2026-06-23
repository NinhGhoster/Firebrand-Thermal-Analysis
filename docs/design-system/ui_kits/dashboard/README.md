# Firebrand Thermal Analysis — Dashboard UI Kit

High-fidelity React recreation of the **redesigned** Firebrand Thermal Analysis dashboard. This is a click-through prototype intended to evaluate the new visual system end-to-end; the underlying logic (file loading, FLIR SDK, OpenCV) is mocked.

## What's in here

- `index.html` — interactive prototype. Open this file. The window mimics the actual app at 1280×800.
- `App.jsx` — top-level composition.
- `SidePanel.jsx` — right-hand telemetry stack (Data Source / Visualisation / Parameters / Export / Footer).
- `Canvas.jsx` — main thermal viewport including ROI overlay, tracking dot, hover readout.
- `ColorBar.jsx` — vertical temperature gradient legend.
- `ControlPod.jsx` — bottom pinned transport bar (slider + Play/Prev/Next + status).
- `Pod.jsx`, `Button.jsx`, `Input.jsx`, `Logo.jsx` — primitives.

## Interactions worth trying

- Press **Play** or hit `Space` to start playback (status & frame counter tick).
- Click any colormap row in **Visualisation** to swap the canvas tint live.
- Drag the scrub slider — the frame counter updates and the tracking dot moves.
- Click **Auto Target Fuel** to draw a fresh ROI.
- Click **Apply** buttons to flash the status line.
- The whole UI is keyboard-navigable; focus rings use the ember accent.

## What's different from the source app

| Area | Source | Redesign |
|---|---|---|
| Accent | Generic Tailwind blue `#3B82F6` | Ember Orange `#FF6B1A` sampled from Inferno |
| Surfaces | Cool slate (`#0F172A` / `#1E293B`) | Warm-dark (`#0B0D10` / `#1C2026`) |
| Section headers | Blue, all on one line | Ember, with breathing room |
| Status line | Amber on the right of the pod | Amber, monospace, with crossfade on update |
| Transport | Text buttons (`Play`, `<`, `>`) | Stroke icons (Lucide) + text |
| Border | None | 1px hairline on every pod |
| Logo | AI-generated raster | Type-based wordmark + inferno gradient mark |
| Icons | None | Lucide stroke icons (16/20px) — flagged as new |

## Loading the prototype

Open `index.html` directly. React 18 + Babel are loaded from a pinned CDN. There is no build step.
