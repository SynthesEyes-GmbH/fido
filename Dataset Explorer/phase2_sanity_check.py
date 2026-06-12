"""
Project an iOCT microscope Volume onto the Stereo-Left view.

Pipeline
--------
1. Load every slice PNG from
       <ROOT>/Scenario_0<SCENARIO>/iOCT Microscope/Volume/<VOL>/
   and stack them into a 3D array.
2. Average projection along the volume's last axis -> 2D en-face image.
3. Load the affine from
       <ROOT>/Scenario_0<SCENARIO>/Numerical/<VOL>.json  ->  ["Ground Truth"]["Phase 2"]
   (a 3x3 matrix; bottom row is [0, 0, 1]).
4. Warp the projection with that affine into the stereo frame.
5. Overlay (augment) the warped projection on
       <ROOT>/Scenario_0<SCENARIO>/Stereo Left/<VOL>/microscope.png
   and save the result.

About the affine coordinate convention
--------------------------------------
The "Phase 2" matrix maps *normalized* [-1, 1] image coordinates (origin at the
image centre) to *pixel* coordinates in the 1024x1024 stereo image. This was
verified against the data: M @ [0, 0, 1] == translation == the centre of the
"iOCT Microscope Crosshair" keypoints, and the linear part's scale equals half
the crosshair square's side length. OCT_SRC_POINTS defines the points (the OCT
square's edge midpoints) that the affine transforms onto those crosshair tips.
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = r"D:\FIDO Challenge\Dataset\Phase 2"
SCENARIO = 1            # scenario index -> Scenario_02
VOL = "00000"           # volume index / folder to START on (Enter advances)
ALPHA = 0.35            # overlay opacity

import json
import cv2
import numpy as np

from draw_utils import compose_phase2_image, overlay
from pathlib import Path



# The projection points the affine transforms, in normalized [-1, 1] coords
# (centre origin). These are the OCT square's edge midpoints; the affine maps
# them onto the four "iOCT Microscope Crosshair" tips in the stereo image.
OCT_SRC_POINTS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.float64)




def interpolate_slices(volume: np.ndarray, n_out: int) -> np.ndarray:
    """Linearly interpolate `volume` along axis 0 from N slices up to `n_out`.

    Maps each output index to a fractional position in the input stack and
    blends the two bracketing slices. Returns a uint8 array of shape
    (n_out, H, W).
    """
    n_in = volume.shape[0]
    vol_f = volume.astype(np.float32)

    # Output index i -> fractional source position over [0, n_in - 1].
    src = np.linspace(0.0, n_in - 1, n_out)
    lo = np.floor(src).astype(np.int64)
    hi = np.minimum(lo + 1, n_in - 1)
    frac = (src - lo).astype(np.float32)[:, None, None]

    out = (1.0 - frac) * vol_f[lo] + frac * vol_f[hi]
    return np.clip(out, 0, 255).astype(np.uint8)


def load_volume(volume_dir: Path) -> np.ndarray:
    """Load slice PNGs in `volume_dir`, keep every other one, and interpolate back.

    Only the even-indexed slices (0, 2, 4, ..., N-2) are read; the full
    N-slice stack is then reconstructed by linear interpolation along the
    slice axis, yielding a (N, H, W) array.
    """
    slice_paths = sorted(p for p in volume_dir.glob("*.png") if p.is_file())
    if not slice_paths:
        raise FileNotFoundError(f"No slice PNGs found in {volume_dir}")

    n_total = len(slice_paths)

    slices = []
    for p in slice_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Failed to read slice: {p}")
        slices.append(img)

    shapes = {s.shape for s in slices}
    if len(shapes) != 1:
        raise ValueError(f"Slices have inconsistent shapes: {shapes}")

    volume = np.stack(slices, axis=0)
    volume = interpolate_slices(volume, n_total)  # back to (n_total, H, W)
    print(f"Loaded {len(slices)}/{n_total} slices -> interpolated volume "
          f"{volume.shape} dtype={volume.dtype}")
    return volume


def average_projection(volume: np.ndarray, axis: int = -1) -> np.ndarray:
    """Axial rojection of the volume along `axis`, returned as uint8."""

    proj = np.mean(volume, axis=axis)
    proj -= proj.min()
    proj /= proj.max()
    proj *= 255
    proj = proj.astype(np.uint8) 
    return proj


def bscan_montage(volume: np.ndarray, rows: int = 11, cols: int = 12) -> np.ndarray:
    """Tile the volume's B-scan slices into a rows x cols grid.

    With 128 slices and an 11x12 = 132 grid, the last 4 cells stay black.
    """
    n, h, w = volume.shape
    canvas = np.zeros((rows * h, cols * w), dtype=volume.dtype)
    for i in range(min(n, rows * cols)):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = volume[i]
    return canvas


def load_affine(json_path: Path) -> np.ndarray:
    """Load ['Ground Truth']['Phase 2'] as a 3x3 affine matrix."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mat = data["Ground Truth"]["Phase 2"]
    if mat is None:
        raise ValueError(f"['Ground Truth']['Phase 2'] is null in {json_path}")
    M = np.asarray(mat, dtype=np.float64)
    if M.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 affine, got {M.shape}")
    print(f"Affine loaded from {json_path.name}:\n{M}")
    return M


def build_warp_matrix(M: np.ndarray, proj_shape) -> np.ndarray:
    """
    Return the 2x3 matrix cv2.warpAffine should apply to the projection pixels.

    OCT_SRC_POINTS (normalized [-1,1], centre origin) are the points the affine
    transforms. We map them to projection PIXEL coords (source) and to stereo
    PIXEL coords via M (destination), then solve for the affine pixel->pixel.
    """
    
    h, w = proj_shape[:2]
    c = np.array([w / 2.0, h / 2.0])                       # (x, y) centre
    src_px = (OCT_SRC_POINTS * c + c).astype(np.float32)   # normalized -> proj px
    pts_h = np.hstack([OCT_SRC_POINTS, np.ones((len(OCT_SRC_POINTS), 1))])
    dst_px = (M @ pts_h.T).T[:, :2].astype(np.float32)     # -> stereo px
    warp, _ = cv2.estimateAffine2D(src_px, dst_px)
    return warp


def process_vol(root: Path, scen: str, vol: str):
    """Build the warped projection and the overlay for one VOL."""
    volume_dir = root / scen / "iOCT Microscope" / "Volume" / vol
    json_path = root / scen / "Numerical" / f"{vol}.json"
    stereo_path = root / scen / "Stereo Left" / vol / "microscope.png"

    # 1-2. Load volume, project, and tile the B-scans.
    volume = load_volume(volume_dir)
    proj = average_projection(volume, axis=1)
    grid = bscan_montage(volume)

    # 3. Load affine.
    M = load_affine(json_path)

    # 4. Warp the projection into the stereo frame.
    stereo = cv2.imread(str(stereo_path), cv2.IMREAD_COLOR)
    if stereo is None:
        raise IOError(f"Failed to read stereo image: {stereo_path}")
    out_h, out_w = stereo.shape[:2]

    warp_M = build_warp_matrix(M, proj.shape)
    print(f"warpAffine matrix (proj px -> stereo px):\n{warp_M}")
    warped = cv2.warpAffine(proj, warp_M, (out_w, out_h),
                            flags=cv2.INTER_LINEAR, borderValue=0)

    # 5. Overlay / augment.
    result = overlay(stereo, warped, alpha=ALPHA)
    return warped, result, grid


def list_vols(root: Path, scen: str) -> list:
    """Sorted VOL folder names available under this scenario's Volume dir."""
    volume_root = root / scen / "iOCT Microscope" / "Volume"
    return sorted(p.name for p in volume_root.iterdir() if p.is_dir())


def main():
    root = Path(ROOT)
    scen = f"Scenario_{SCENARIO:02d}"

    vols = list_vols(root, scen)
    if VOL in vols:
        idx = vols.index(VOL)
    else:
        print(f"VOL {VOL!r} not found; starting at first ({vols[0]}).")
        idx = 0

    win = "OCT overlay + warped projection"
    print("Enter / Space / Right = next   |   Left / p = previous   |   Esc / q = quit")

    while 0 <= idx < len(vols):
        vol = vols[idx]
        print(f"\n=== [{idx + 1}/{len(vols)}] {scen} / {vol} ===")
        try:
            warped, result, grid = process_vol(root, scen, vol)
        except (FileNotFoundError, IOError, ValueError) as e:
            print(f"  skipping {vol}: {e}")
            idx += 1
            continue

        view = compose_phase2_image(result, warped, grid)
        cv2.imshow(win, view)
        cv2.setWindowTitle(win, f"{scen} / {vol}  [{idx + 1}/{len(vols)}]")

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
