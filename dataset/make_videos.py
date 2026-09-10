#!/usr/bin/env python3
"""
Dataset video generator for FIDO Task 1 scenarios.

Usage:
    python make_videos.py --data <scenario_folder>
    python make_videos.py --data <scenario_folder> --out <output_folder>
    python make_videos.py --data <scenario_folder> --preview-ts 00500
    python make_videos.py --data <scenario_folder> --full

Arguments:
    --data          Path to a scenario folder (must contain "Stereo Left",
                    "iOCT Microscope", and "Numerical" subfolders).
    --out           Where to write outputs (default: alongside this script).
    --full          Encode all videos after saving the preview.
    --fps           Video frame rate (default: 25).
    --preview-ts    Timestamp to render for the preview image (default: middle frame).

Outputs (named after the scenario folder):
    <scenario>_preview.png
    <scenario>_stereo_real.mp4    — microscope + OCT crosshair lines
    <scenario>_stereo_seg.mp4     — composite segmentation map
    <scenario>_keypoints.mp4      — microscope + vessel GT points + cannula
    <scenario>_oct0_real.mp4      — B-scan 00 raw
    <scenario>_oct1_real.mp4      — B-scan 01 raw
    <scenario>_oct0_seg.mp4       — B-scan 00 segmentation
    <scenario>_oct1_seg.mp4       — B-scan 01 segmentation
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

def _hex(h: str):
    r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    return (b, g, r)  # BGR

SEG_COLORS_STEREO = {
    "arteriesorveins": _hex("#0096A4"),
    "ilm":             _hex("#1D1D1C"),
    "cannula":         _hex("#01D7A1"),
    "endoilluminator": _hex("#D81C3F"),
}
SEG_ORDER = ["ilm", "arteriesorveins", "cannula", "endoilluminator", "forceps"]

OCT_SEG_COLORS = {
    0:  (0, 0, 0),
    1:  _hex("#1D1D1C"),
    2:  (0x20, 0x50, 0x70),
    3:  _hex("#0096A4"),
    11: _hex("#01D7A1"),
    13: _hex("#D81C3F"),
}

CROSSHAIR_COLORS = [
    (194, 3, 252),      # pink-magenta — line 0
    _hex("#86EFE3"),    # #86EFE3      — line 1
]

CANNULA_COLOR = _hex("#01D7A1")

# ---------------------------------------------------------------------------
# Dataset accessors  (all path-agnostic; callers pass roots)
# ---------------------------------------------------------------------------

def _timestamps(stereo_root: Path) -> list[str]:
    return sorted(p.name for p in stereo_root.iterdir() if p.is_dir())


def load_microscope(stereo_root: Path, ts: str) -> np.ndarray:
    with Image.open(stereo_root / ts / "microscope.png") as img:
        return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def load_stereo_seg(stereo_root: Path, ts: str) -> np.ndarray:
    seg_dir = stereo_root / ts / "Segmentation"
    sample = load_microscope(stereo_root, ts)
    canvas = np.zeros_like(sample)
    for name in SEG_ORDER:
        path = seg_dir / f"{name}.png"
        if not path.exists():
            continue
        mask = np.array(Image.open(path).convert("L")) > 128
        color = SEG_COLORS_STEREO.get(name)
        if color is not None:
            canvas[mask] = color
    return canvas


def load_crosshair(num_root: Path, ts: str):
    with open(num_root / f"{ts}.json") as f:
        data = json.load(f)
    ch = data["Keypoints"]["iOCT Microscope Crosshair"]
    return [
        (ch["Start 0"], ch["End 0"]),
        (ch["Start 1"], ch["End 1"]),
    ]


def load_keypoints(num_root: Path, ts: str) -> dict:
    with open(num_root / f"{ts}.json") as f:
        data = json.load(f)
    vasc    = data["Keypoints"]["Vasculature"]
    cannula = data["Keypoints"]["Cannula SRI"]
    return {
        "vessels":       [v for k, v in vasc.items() if k.startswith(("V_", "A_"))],
        "cannula_tip":   cannula["Tip"],
        "cannula_start": cannula["Start"],
    }


def load_oct_slice(oct_root: Path, ts: str, i: int) -> np.ndarray:
    with Image.open(oct_root / ts / f"{i:02d}.png") as img:
        gray = np.array(img.convert("L"))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def load_oct_seg_slice(oct_root: Path, ts: str, i: int) -> np.ndarray:
    with Image.open(oct_root / ts / "Segmentation" / f"{i:02d}.png") as img:
        labels = np.array(img.convert("L"))
    canvas = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label, color in OCT_SEG_COLORS.items():
        canvas[labels == label] = color
    return canvas

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_crosshair(bgr: np.ndarray, lines) -> np.ndarray:
    out = bgr.copy()
    for (start, end), color in zip(lines, CROSSHAIR_COLORS):
        p0 = (int(round(start[0])), int(round(start[1])))
        p1 = (int(round(end[0])), int(round(end[1])))
        cv2.line(out, p0, p1, color, 2, cv2.LINE_AA)
        cv2.circle(out, p0, 5, color, -1, cv2.LINE_AA)
    return out


def _pos_color(pt, cx: float, cy: float):
    angle = (np.degrees(np.arctan2(pt[1] - cy, pt[0] - cx)) % 360) / 360.0
    hsv = np.array([[[int(angle * 179), 220, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def draw_keypoints(bgr: np.ndarray, kp: dict) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    for pt in kp["vessels"]:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= x < w and 0 <= y < h):
            continue
        c = _pos_color(pt, cx, cy)
        cv2.circle(out, (x, y), 3, c, -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 7, c,  1, cv2.LINE_AA)

    tx, ty = int(round(kp["cannula_tip"][0])),   int(round(kp["cannula_tip"][1]))
    sx, sy = int(round(kp["cannula_start"][0])), int(round(kp["cannula_start"][1]))
    cv2.line(out, (sx, sy), (tx, ty), CANNULA_COLOR, 2, cv2.LINE_AA)
    cv2.circle(out, (sx, sy), 8, CANNULA_COLOR, 2, cv2.LINE_AA)
    h2 = 8
    cv2.rectangle(out, (tx - h2, ty - h2), (tx + h2, ty + h2), CANNULA_COLOR, 2, cv2.LINE_AA)
    cv2.circle(out, (tx, ty), 2, CANNULA_COLOR, -1, cv2.LINE_AA)
    return out

# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
    return out


def build_preview(stereo_root: Path, oct_root: Path, num_root: Path, ts: str) -> np.ndarray:
    mic    = load_microscope(stereo_root, ts)
    lines  = load_crosshair(num_root, ts)
    sr     = draw_crosshair(mic, lines)
    ss     = draw_crosshair(load_stereo_seg(stereo_root, ts), lines)

    cell_h, cell_w = sr.shape[:2]
    top = np.hstack([_label(sr, "Stereo Real + OCT lines"),
                     _label(ss, "Stereo Segmentation")])

    row_w = top.shape[1]
    oct_cell_w = row_w // 2

    def _fit_w(img):
        ih, iw = img.shape[:2]
        nw = oct_cell_w
        nh = int(ih * nw / iw)
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    or0 = _fit_w(np.hstack([load_oct_slice(oct_root, ts, 0),
                             load_oct_slice(oct_root, ts, 1)]))
    os0 = _fit_w(np.hstack([load_oct_seg_slice(oct_root, ts, 0),
                             load_oct_seg_slice(oct_root, ts, 1)]))

    bh = max(or0.shape[0], os0.shape[0])
    def _pad_h(img):
        if img.shape[0] == bh: return img
        return np.vstack([img, np.zeros((bh - img.shape[0], img.shape[1], 3), np.uint8)])

    bottom = np.hstack([_label(_pad_h(or0), "OCT Real (both slices)"),
                        _label(_pad_h(os0), "OCT Segmentation (both slices)")])

    preview = np.vstack([top, bottom])
    ph, pw = preview.shape[:2]
    scale = min(1.0, 1800 / max(ph, pw))
    if scale < 1.0:
        preview = cv2.resize(preview, (int(pw * scale), int(ph * scale)),
                             interpolation=cv2.INTER_AREA)
    return preview

# ---------------------------------------------------------------------------
# Video encoding
# ---------------------------------------------------------------------------

def _writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    h, w = frame.shape[:2]
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


def encode_all(stereo_root: Path, oct_root: Path, num_root: Path,
               timestamps: list[str], out_dir: Path, prefix: str, fps: float):
    writers: dict[str, cv2.VideoWriter] = {}

    def get_w(key, frame):
        if key not in writers:
            writers[key] = _writer(out_dir / f"{prefix}_{key}.mp4", frame, fps)
        return writers[key]

    total = len(timestamps)
    try:
        for idx, ts in enumerate(timestamps):
            if idx % 200 == 0:
                print(f"  [{idx+1}/{total}] {ts}")

            mic   = load_microscope(stereo_root, ts)
            lines = load_crosshair(num_root, ts)

            get_w("stereo_real", mic).write(draw_crosshair(mic, lines))
            get_w("stereo_seg",  mic).write(draw_crosshair(load_stereo_seg(stereo_root, ts), lines))
            get_w("keypoints",   mic).write(draw_keypoints(mic, load_keypoints(num_root, ts)))

            for i in range(2):
                get_w(f"oct{i}_real", load_oct_slice(oct_root, ts, i)).write(
                    load_oct_slice(oct_root, ts, i))
                get_w(f"oct{i}_seg",  load_oct_seg_slice(oct_root, ts, i)).write(
                    load_oct_seg_slice(oct_root, ts, i))

    finally:
        for w in writers.values():
            w.release()
        saved = "\n  ".join(f"{prefix}_{k}.mp4" for k in writers)
        print(f"\nSaved to {out_dir}:\n  {saved}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate annotated videos from a FIDO Task 1 scenario folder.")
    parser.add_argument("--data", required=True,
                        help="Scenario folder (contains 'Stereo Left', 'iOCT Microscope', 'Numerical')")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: alongside this script)")
    parser.add_argument("--full", action="store_true",
                        help="Encode all 7 videos (default: preview only)")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Video frame rate (default: 25)")
    parser.add_argument("--preview-ts", default=None,
                        help="Timestamp for the preview frame (default: middle frame)")
    args = parser.parse_args()

    scenario = Path(args.data).expanduser().resolve()
    if not scenario.is_dir():
        parser.error(f"--data not found: {scenario}")

    stereo_root = scenario / "Stereo Left"
    oct_root    = scenario / "iOCT Microscope" / "Bscan"
    num_root    = scenario / "Numerical"
    for d, name in [(stereo_root, "Stereo Left"),
                    (oct_root,    "iOCT Microscope/Bscan"),
                    (num_root,    "Numerical")]:
        if not d.is_dir():
            parser.error(f"Expected subfolder not found: {d}  (looked for '{name}')")

    out_dir = Path(args.out).expanduser().resolve() if args.out else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamps = _timestamps(stereo_root)
    if not timestamps:
        parser.error(f"No timestamp folders found under {stereo_root}")

    prefix    = scenario.name
    preview_ts = args.preview_ts or timestamps[len(timestamps) // 2]

    print(f"Scenario : {scenario}")
    print(f"Frames   : {len(timestamps)}  ({timestamps[0]} → {timestamps[-1]})")
    print(f"Output   : {out_dir}")
    print(f"Preview  : timestamp {preview_ts}\n")

    preview = build_preview(stereo_root, oct_root, num_root, preview_ts)
    preview_path = out_dir / f"{prefix}_preview.png"
    cv2.imwrite(str(preview_path), preview)
    print(f"Preview saved → {preview_path}")

    if args.full:
        print(f"\nEncoding {len(timestamps)} frames × 7 videos …")
        encode_all(stereo_root, oct_root, num_root, timestamps, out_dir, prefix, args.fps)


if __name__ == "__main__":
    main()
