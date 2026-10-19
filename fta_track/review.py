"""Blind yes/no review of tracks, to measure how often the pipeline is right.

`make_batch` samples tracks from one recording's results, renders each as a
short clip (a sprite sheet) plus a context thumbnail, and appends them to a
batch directory. `write_page` writes a self-contained index.html: the reviewer
sees one clip at a time, in shuffled order, without the pipeline's decision,
and answers y (firebrand leaving the sample), n (not), u (unsure). Verdicts
persist in the browser and are downloaded as verdicts.json.

`score` compares verdicts with the pipeline's decisions: the share of counted
firebrands that are real (precision), and the share of rejected tracks that
were real firebrands (what the gates threw away), per recording and per
rejection reason.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import random
from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np

# kept (hidden) with each clip, so verdicts can be turned into threshold curves
FEATURES = ("n_det", "n_cold", "n_evidence", "max_contrast_c", "displacement_m", "straightness",
            "peak_c", "mean_speed_mps", "duration_s", "mean_area_px", "area_jump", "edge_grad")
SPRITE_COLS = 12
TILE = 192            # clip frame size in the sprite, px
MAX_CLIP_FRAMES = 96
PAD_FRAMES = 8
VMIN, VMAX = 20.0, 800.0


def _read_tracks(results_dir: str) -> List[Dict]:
    with open(os.path.join(results_dir, "tracks.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    pts = defaultdict(list)
    extra = defaultdict(list)
    with gzip.open(os.path.join(results_dir, "track_points.csv.gz"), "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            tid = int(r["track_id"])
            pts[tid].append((int(r["frame"]), float(r["x_px"]), float(r["y_px"])))
            extra[tid].append((float(r["max_c"]), float(r["contrast_c"]) if r["contrast_c"] else None))
    hf_path = os.path.join(results_dir, "hot_fraction.npz")
    hot = np.load(hf_path)["hot_frac"].astype(np.float32) if os.path.exists(hf_path) else None
    for r in rows:
        tid = int(r["track_id"])
        r["track_id"] = tid
        r["n_det"] = int(r["n_det"])
        order = sorted(range(len(pts[tid])), key=lambda k: pts[tid][k][0])
        r["points"] = [pts[tid][k] for k in order]
        # everything needed to re-decide this track under any threshold, without the results dir
        r["point_data"] = []
        for k in order:
            f, x, y = pts[tid][k]
            mx, con = extra[tid][k]
            hf = None
            if hot is not None:
                hf = round(float(hot[min(hot.shape[0] - 1, max(0, int(round(y)))),
                                     min(hot.shape[1] - 1, max(0, int(round(x))))]), 4)
            r["point_data"].append([f, round(x, 2), round(y, 2), mx, con, hf])
    return rows


def _num(v):
    if v in (None, ""):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return int(f) if f.is_integer() and "." not in str(v) else f


def _colour(frame: np.ndarray) -> np.ndarray:
    g = ((np.clip(frame, VMIN, VMAX) - VMIN) / (VMAX - VMIN) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)


def _crop_rect(points, W, H, pad=48, min_side=96):
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    side = max(x1 - x0, y1 - y0, min_side)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    side = int(round(side))
    x0 = min(max(0, x0), max(0, W - side))
    y0 = min(max(0, y0), max(0, H - side))
    return x0, y0, min(side, W), min(side, H)


def _render(var, track: Dict, n_frames: int, out_png: str, out_ctx: str) -> Dict:
    H, W = var.shape[1], var.shape[2]
    f0 = max(0, track["points"][0][0] - PAD_FRAMES)
    f1 = min(n_frames - 1, track["points"][-1][0] + PAD_FRAMES)
    stride = max(1, math.ceil((f1 - f0 + 1) / MAX_CLIP_FRAMES))
    frames = list(range(f0, f1 + 1, stride))
    x0, y0, cw, ch = _crop_rect(track["points"], W, H)
    scale = TILE / max(cw, ch)
    at = {f: (x, y) for f, x, y in track["points"]}

    block = np.asarray(var[f0:f1 + 1, y0:y0 + ch, x0:x0 + cw], dtype=np.float32)
    rows = math.ceil(len(frames) / SPRITE_COLS)
    sprite = np.zeros((rows * TILE, SPRITE_COLS * TILE, 3), dtype=np.uint8)
    marks = []
    for k, f in enumerate(frames):
        tile = cv2.resize(_colour(block[f - f0]), (int(round(cw * scale)), int(round(ch * scale))),
                          interpolation=cv2.INTER_NEAREST)
        r, c = divmod(k, SPRITE_COLS)
        sprite[r * TILE:r * TILE + tile.shape[0], c * TILE:c * TILE + tile.shape[1]] = tile
        p = at.get(f)
        marks.append(None if p is None else [round((p[0] - x0) * scale, 1), round((p[1] - y0) * scale, 1)])
    cv2.imwrite(out_png, sprite, [cv2.IMWRITE_PNG_COMPRESSION, 6])

    mid = track["points"][len(track["points"]) // 2][0]
    ctx = _colour(np.asarray(var[mid], dtype=np.float32))
    poly = np.array([[int(x), int(y)] for _, x, y in track["points"]], dtype=np.int32)
    cv2.polylines(ctx, [poly], False, (255, 255, 0), 2)
    cv2.rectangle(ctx, (x0, y0), (x0 + cw, y0 + ch), (255, 255, 255), 2)
    cv2.imwrite(out_ctx, cv2.resize(ctx, (W * 3 // 8, H * 3 // 8), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"frames": frames, "marks": marks, "n_tiles": len(frames), "tile": TILE,
            "cols": SPRITE_COLS, "crop_px": [x0, y0, cw, ch]}


def make_batch(results_dir: str, nc_path: str, batch_dir: str, label: str,
               n_firebrand: int = 20, n_rejected: int = 15, min_det: int = 4, seed: int = 0) -> int:
    import netCDF4 as nc

    tracks = _read_tracks(results_dir)
    rng = random.Random(f"{seed}:{label}")
    fb = [t for t in tracks if t["is_firebrand"] == "1"]
    pick = rng.sample(fb, min(n_firebrand, len(fb)))
    # Rejected tracks, spread evenly over the rejection reasons, so the borderline
    # ones (low contrast, little movement) are judged and not just obvious junk.
    by_reason = defaultdict(list)
    for t in tracks:
        if t["is_firebrand"] != "1" and t["n_det"] >= min_det:
            by_reason[t["reject_reason"]].append(t)
    reasons = sorted(by_reason)
    quota = {r: 0 for r in reasons}
    left = n_rejected
    while left > 0 and any(quota[r] < len(by_reason[r]) for r in reasons):
        for r in reasons:
            if left > 0 and quota[r] < len(by_reason[r]):
                quota[r] += 1
                left -= 1
    for r in reasons:
        pick += rng.sample(by_reason[r], quota[r])

    clips_dir = os.path.join(batch_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    man_path = os.path.join(batch_dir, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {"clips": []}
    manifest["clips"] = [c for c in manifest["clips"] if c["label"] != label]

    ds = nc.Dataset(nc_path, "r")
    try:
        var = ds.variables["temperature"]
        var.set_auto_maskandscale(False)
        n_frames = var.shape[0]
        fps = float(ds.getncattr("frame_rate")) if "frame_rate" in ds.ncattrs() else 30.0
        safe = label.replace("/", "__")
        for t in pick:
            cid = f"{safe}__{t['track_id']}"
            info = _render(var, t, n_frames, os.path.join(clips_dir, cid + ".png"),
                           os.path.join(clips_dir, cid + "_ctx.jpg"))
            manifest["clips"].append({
                "id": cid, "label": label, "track_id": t["track_id"],
                # hidden from the reviewer; used only by score()
                "pipeline_firebrand": t["is_firebrand"] == "1",
                "reject_reason": t["reject_reason"],
                "features": {k: _num(t.get(k)) for k in FEATURES},
                "point_data": t["point_data"],   # [frame, x, y, max_c, contrast_c, hot_frac]
                "fps": fps, **info,
            })
    finally:
        ds.close()
    with open(man_path, "w") as fh:
        json.dump(manifest, fh)
    write_page(batch_dir)
    return len(pick)


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Firebrand track review</title>
<style>
:root { --bg:#14161a; --fg:#e8e6e1; --mut:#9a9890; --yes:#3fb27f; --no:#e5534b; --uns:#d4a72c; --card:#1d2026; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; }
main { max-width:980px; margin:0 auto; padding:16px; }
h1 { font-size:18px; margin:0 0 4px; }
.sub { color:var(--mut); margin:0 0 14px; }
.row { display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }
.card { background:var(--card); border-radius:8px; padding:12px; }
canvas { display:block; image-rendering:pixelated; width:384px; height:384px; max-width:100%; background:#000; border-radius:4px; }
#ctx { width:384px; max-width:100%; border-radius:4px; display:block; }
.q { font-size:16px; margin:14px 0 8px; }
.btns button { font:inherit; padding:10px 18px; margin:0 8px 8px 0; border:0; border-radius:6px; cursor:pointer; color:#111; font-weight:600; }
.y { background:var(--yes); } .n { background:var(--no); } .u { background:var(--uns); }
.nav button { font:inherit; padding:6px 12px; margin-right:6px; background:#2c3038; color:var(--fg); border:0; border-radius:6px; cursor:pointer; }
.state { color:var(--mut); font-size:13px; }
.done { font-weight:700; }
kbd { background:#2c3038; border-radius:4px; padding:1px 6px; font-size:12px; }
.bar { height:6px; background:#2c3038; border-radius:3px; overflow:hidden; margin:8px 0 14px; }
.bar > div { height:100%; background:var(--yes); width:0; }
</style></head><body><main>
<h1>Is this a firebrand leaving the sample?</h1>
<p class="sub">Each clip follows one track the pipeline found. Answer for the object the cyan ring follows.
<kbd>y</kbd> firebrand · <kbd>n</kbd> not (flame, hot gas, glare, burner, noise) · <kbd>u</kbd> unsure ·
<kbd>←</kbd><kbd>→</kbd> move · <kbd>space</kbd> replay · <kbd>m</kbd> ring on/off</p>
<div class="bar"><div id="prog"></div></div>
<div class="row">
  <div class="card"><canvas id="cv" width="192" height="192"></canvas></div>
  <div class="card"><img id="ctx" alt="whole frame at the middle of the track: white box = clip area, cyan line = path">
    <p class="state" id="meta"></p></div>
</div>
<p class="q" id="ans"></p>
<div class="btns"><button class="y" data-v="y">Yes · firebrand</button><button class="n" data-v="n">No</button><button class="u" data-v="u">Unsure</button></div>
<div class="nav"><button id="prev">← Previous</button><button id="next">Next →</button>
<button id="dl">Download verdicts.json</button> <span class="state" id="count"></span></div>
</main>
<script>
const M = __MANIFEST__;
const KEY = "fta-review-" + __BATCH__;
let order = M.clips.map((c, i) => i);
(function shuffle(a, s) { for (let i = a.length - 1; i > 0; i--) { s = (s * 9301 + 49297) % 233280; const j = Math.floor(s / 233280 * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } })(order, 12345);
let V = {}; try { V = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let pos = Math.max(0, order.findIndex(i => !(M.clips[i].id in V))); if (pos < 0) pos = 0;
let showMark = true, timer = null, img = new Image();
const cv = document.getElementById("cv"), g = cv.getContext("2d");
function save() { try { localStorage.setItem(KEY, JSON.stringify(V)); } catch (e) {} }
function show() {
  const c = M.clips[order[pos]];
  img = new Image(); img.src = "clips/" + c.id + ".png"; img.onload = play;
  document.getElementById("ctx").src = "clips/" + c.id + "_ctx.jpg";
  document.getElementById("meta").textContent = "Clip " + (pos + 1) + " of " + order.length + " · " + c.frames.length + " frames at " + c.fps + " fps (" + (c.frames.length * (c.frames[1] - c.frames[0] || 1) / c.fps).toFixed(1) + " s)";
  const v = V[c.id];
  document.getElementById("ans").innerHTML = v ? "Your answer: <span class='done'>" + {y:"Yes",n:"No",u:"Unsure"}[v] + "</span>" : "No answer yet";
  const n = Object.keys(V).length;
  document.getElementById("count").textContent = n + " of " + order.length + " answered";
  document.getElementById("prog").style.width = (100 * n / order.length) + "%";
}
function play() {
  clearInterval(timer);
  const c = M.clips[order[pos]]; let k = 0;
  const dt = Math.max(30, 1000 / c.fps * (c.frames[1] - c.frames[0] || 1));
  timer = setInterval(() => {
    const r = Math.floor(k / c.cols), q = k % c.cols;
    g.clearRect(0, 0, 192, 192);
    g.drawImage(img, q * c.tile, r * c.tile, c.tile, c.tile, 0, 0, 192, 192);
    const m = c.marks[k];
    if (showMark && m) { g.strokeStyle = "#00e5ff"; g.lineWidth = 1.5; g.beginPath(); g.arc(m[0], m[1], 9, 0, 2 * Math.PI); g.stroke(); }
    k++; if (k >= c.n_tiles) k = 0;
  }, dt);
}
function answer(v) { V[M.clips[order[pos]].id] = v; save(); if (pos < order.length - 1) pos++; show(); }
document.querySelectorAll(".btns button").forEach(b => b.onclick = () => answer(b.dataset.v));
document.getElementById("prev").onclick = () => { if (pos > 0) pos--; show(); };
document.getElementById("next").onclick = () => { if (pos < order.length - 1) pos++; show(); };
document.getElementById("dl").onclick = () => {
  const blob = new Blob([JSON.stringify({batch: __BATCH__, verdicts: V}, null, 1)], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "verdicts.json"; a.click();
};
document.addEventListener("keydown", e => {
  if (e.key === "y" || e.key === "n" || e.key === "u") answer(e.key);
  else if (e.key === "ArrowLeft") { if (pos > 0) pos--; show(); }
  else if (e.key === "ArrowRight") { if (pos < order.length - 1) pos++; show(); }
  else if (e.key === " ") { e.preventDefault(); play(); }
  else if (e.key === "m") showMark = !showMark;
});
show();
</script></body></html>
"""


def merge_batches(out_dir: str, parts: List[str]) -> int:
    """Combine per-recording batches (built in parallel) into one review page."""
    import shutil

    os.makedirs(os.path.join(out_dir, "clips"), exist_ok=True)
    clips, seen = [], set()
    for part in parts:
        man = json.load(open(os.path.join(part, "manifest.json")))
        for c in man["clips"]:
            if c["id"] in seen:
                continue
            seen.add(c["id"])
            for suffix in (".png", "_ctx.jpg"):
                shutil.copy2(os.path.join(part, "clips", c["id"] + suffix),
                             os.path.join(out_dir, "clips", c["id"] + suffix))
            clips.append(c)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump({"clips": clips}, fh)
    write_page(out_dir)
    return len(clips)


def write_page(batch_dir: str) -> str:
    man = json.load(open(os.path.join(batch_dir, "manifest.json")))
    # the reviewer's page carries no pipeline decision
    public = {"clips": [{k: v for k, v in c.items()
                         if k not in ("pipeline_firebrand", "reject_reason", "label", "track_id", "features",
                                      "point_data")}
                        for c in man["clips"]]}
    html = _PAGE.replace("__MANIFEST__", json.dumps(public)).replace(
        "__BATCH__", json.dumps(os.path.basename(os.path.abspath(batch_dir))))
    path = os.path.join(batch_dir, "index.html")
    with open(path, "w") as fh:
        fh.write(html)
    return path


def score(batch_dir: str, verdicts_path: str) -> Dict:
    man = {c["id"]: c for c in json.load(open(os.path.join(batch_dir, "manifest.json")))["clips"]}
    V = json.load(open(verdicts_path))["verdicts"]

    def tally(rows):
        y = sum(1 for r in rows if r == "y")
        n = sum(1 for r in rows if r == "n")
        u = sum(1 for r in rows if r == "u")
        return {"yes": y, "no": n, "unsure": u, "share_yes": round(y / (y + n), 3) if (y + n) else None}

    fb, rej = defaultdict(list), defaultdict(list)
    by_reason = defaultdict(list)
    for cid, v in V.items():
        c = man.get(cid)
        if c is None:
            continue
        (fb if c["pipeline_firebrand"] else rej)[c["label"]].append(v)
        if not c["pipeline_firebrand"]:
            by_reason[c["reject_reason"]].append(v)
    out = {
        "answered": len(V),
        "precision_of_counted_firebrands": tally([v for vs in fb.values() for v in vs]),
        "real_firebrands_among_rejected": tally([v for vs in rej.values() for v in vs]),
        "per_recording": {lab: {"counted": tally(fb.get(lab, [])), "rejected": tally(rej.get(lab, []))}
                          for lab in sorted(set(fb) | set(rej))},
        "rejected_by_reason": {k: tally(v) for k, v in sorted(by_reason.items())},
    }
    with open(os.path.join(batch_dir, "scores.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return out
