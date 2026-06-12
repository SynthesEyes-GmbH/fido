"""
Show the Phase 1 ground-truth point on the Stereo-Left view, next to its B-scans.

Pipeline
--------
1. Load the stereo image from
       <ROOT>/Scenario_0<SCENARIO>/Stereo Left/<FRAME>/microscope.png   (1024x1024)
2. Load the ground truth from
       <ROOT>/Scenario_0<SCENARIO>/Numerical/<FRAME*10>.json  ->  ["Ground Truth"]["Phase 1"]
   which is [x, y, z]: (x, y) are pixel coords in the stereo image; z is the
   retina distance.
3. Draw a marker at (x, y) on the stereo image.
4. Load the two B-scans from
       <ROOT>/Scenario_0<SCENARIO>/iOCT Microscope/Bscan/<FRAME>/00.png  (512x512)
       <ROOT>/Scenario_0<SCENARIO>/iOCT Microscope/Bscan/<FRAME>/01.png  (512x512)
   and show them next to the stereo image, annotated with "retina distance: {z}".

Naming note
-----------
The Stereo Left / Bscan folders are 1-indexed frame ids (00001, 00002, ...),
but the matching Numerical JSON is named frame_id * 10 (folder 00001 ->
00010.json). FRAME below is the Stereo Left / Bscan folder name.
"""
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = r"D:\FIDO Challenge\Dataset\Phase 1"
SCENARIO = 1            # scenario index -> Scenario_01
FRAME = "00010"         # Stereo Left / Bscan folder to START on (Enter advances)

import json
from pathlib import Path
from constants import GenericLabels

import cv2
import numpy as np

from draw_utils import (
    compose_phase1_image,
    draw_crosshair_lines,
    draw_instrument_point,
    draw_point,
)




def numerical_name(frame: str) -> str:
    return f"{int(frame) * 1:05d}"

def load_phase1_point(json_path: Path):
    """Load ['Ground Truth']['Phase 1'] -> (x, y, z) floats."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pt = data["Ground Truth"]["Phase 1"]
    if pt is None:
        raise ValueError(f"['Ground Truth']['Phase 1'] is null in {json_path}")
    x, y, z = (float(v) for v in pt)
    print(f"Phase 1 point from {json_path.name}: x={x:.2f} y={y:.2f} z={z:.4f}")
    return x, y, z


def load_crosshair_lines(json_path: Path):
    """Load the OCT crosshair lines [(Start 0, End 0), (Start 1, End 1)] or None.

    Each entry is a (start_xy, end_xy) pair in stereo pixel coords. Line i
    corresponds to B-scan i.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ch = data.get("Keypoints", {}).get("iOCT Microscope Crosshair")
    if not ch:
        return None
    try:
        return [(ch["Start 0"], ch["End 0"]), (ch["Start 1"], ch["End 1"])]
    except KeyError:
        return None


def load_bscans(bscan_dir: Path):
    """Load the two B-scans (00.png, 01.png) as BGR images."""
    scans = []
    for name in ("00.png", "01.png"):
        p = bscan_dir / name
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to read B-scan: {p}")
        scans.append(img)
    return scans


# Instrument labels in the B-scan segmentation masks.
INSTRUMENT_LABELS = (
    GenericLabels.Forceps.value,
    GenericLabels.Cannula.value,
    GenericLabels.Endoilluminator.value,
    GenericLabels.InstrumentInOCT.value,
)


def load_bscan_segs(bscan_dir: Path):
    """Load the two B-scan segmentation masks (Segmentation/00.png, 01.png).

    Returns single-channel label maps; an entry is None if the mask is absent.
    """
    seg_dir = bscan_dir / "Segmentation"
    segs = []
    for name in ("00.png", "01.png"):
        seg = cv2.imread(str(seg_dir / name), cv2.IMREAD_UNCHANGED)
        if seg is not None and seg.ndim == 3:
            seg = seg[..., 0]
        segs.append(seg)
    return segs


def lowest_x_instrument_point(seg: np.ndarray):
    """Pixel (x, y) of the instrument-labeled pixel with the smallest x, or None.

    Considers Forceps / Cannula / Endoilluminator / InstrumentInOCT. Ties on x
    are broken by the smallest y (topmost).
    """
    if seg is None:
        return None
    mask = np.isin(seg, INSTRUMENT_LABELS)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    min_x = xs.min()
    y = ys[xs == min_x].min()
    return int(min_x), int(y)



def process_frame(root: Path, scen: str, frame: str) -> np.ndarray:
    """Build the composite view for one frame."""
    stereo_path = root / scen / "Stereo Left" / frame / "microscope.png"
    json_path = root / scen / "Numerical" / f"{numerical_name(frame)}.json"
    bscan_dir = root / scen / "iOCT Microscope" / "Bscan" / frame

    stereo = cv2.imread(str(stereo_path), cv2.IMREAD_COLOR)
    if stereo is None:
        raise IOError(f"Failed to read stereo image: {stereo_path}")

    lines = load_crosshair_lines(json_path)
    marked = draw_crosshair_lines(stereo, lines)
    
    x, y, z = load_phase1_point(json_path)
    marked = draw_point(marked, x, y)

    scans = load_bscans(bscan_dir)
    segs = load_bscan_segs(bscan_dir)
    marked_scans = []
    for i, (sc, seg) in enumerate(zip(scans, segs)):
        pt = lowest_x_instrument_point(seg)
        if pt is not None:
            print(f"  Bscan {i:02d}: lowest-x instrument pixel at {pt}")
        marked_scans.append(draw_instrument_point(sc, pt))

    return compose_phase1_image(marked, marked_scans, z)


def list_frames(root: Path, scen: str) -> list:
    """Sorted frame folder names available under this scenario's Stereo Left dir."""
    stereo_root = root / scen / "Stereo Left"
    return sorted(p.name for p in stereo_root.iterdir() if p.is_dir())


def main():
    root = Path(ROOT)
    scen = f"Scenario_{SCENARIO:02d}"

    frames = list_frames(root, scen)
    if FRAME in frames:
        idx = frames.index(FRAME)
    else:
        print(f"FRAME {FRAME!r} not found; starting at first ({frames[0]}).")
        idx = 0

    win = "phase 1 point + bscans"
    print("Enter / Space / Right = next   |   Left / p = previous   |   Esc / q = quit")

    while 0 <= idx < len(frames):
        frame = frames[idx]
        print(f"\n=== [{idx + 1}/{len(frames)}] {scen} / {frame} ===")
        try:
            view = process_frame(root, scen, frame)
        except (FileNotFoundError, IOError, ValueError) as e:
            print(f"  skipping {frame}: {e}")
            idx += 1
            continue

        cv2.imshow(win, view)
        cv2.setWindowTitle(win, f"{scen} / {frame}  [{idx + 1}/{len(frames)}]")

        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):           # Esc / q -> quit
            break
        elif key in (81, ord("p")):         # Left / p -> previous
            idx = max(0, idx - 1)
        else:                               # Enter / Space / Right / anything -> next
            idx += 1

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
