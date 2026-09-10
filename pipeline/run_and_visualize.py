#!/usr/bin/env python3
"""
Run inference and visualize results for Task 1 or Task 2.

Uses the Codabench bundle's ingestion logic for inference; only the data-loading
paths are adapted for local input formats.

Task 1:
    python run_and_visualize.py --task 1 --bundle <folder> --data <video.mp4>
    python run_and_visualize.py --task 1 --bundle <folder> --data <scenario_folder>

    Saves an annotated MP4 next to the input (or to --out).
    Also shows each frame in an OpenCV window while encoding (waitKey(1)).

Task 2:
    python run_and_visualize.py --task 2 --bundle <folder> --data <data_root>

    Data root must contain Task 2/ with scenario subfolders in the standard layout.
    Saves one annotated PNG per case into --out-dir (default: next to this script).
    Then shows the saved PNGs frame by frame in an OpenCV window.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BUNDLE_ROOT = Path(__file__).parent.parent / "Codabench Bundle"

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

# ---------------------------------------------------------------------------
# Colors  (BGR)
# ---------------------------------------------------------------------------

def _hex(h: str):
    r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    return (b, g, r)

PRED_COLOR = _hex("#03d7a1")
GT_COLOR   = (0, 200, 0)
LINE_COLOR = (0, 200, 200)
PRED_HALF  = 15
CORNER_R   = 4


# ---------------------------------------------------------------------------
# Submission loader (mirrors Codabench ingestion)
# ---------------------------------------------------------------------------

def load_submission(bundle_dir: Path, task: int):
    model_file = f"model_{task - 1}.pth"
    required   = {"inference.py", model_file}
    present    = {p.name for p in bundle_dir.iterdir() if p.is_file()}
    missing    = required - present
    if missing:
        nested = [d for d in bundle_dir.iterdir()
                  if d.is_dir() and not (required - {p.name for p in d.iterdir() if p.is_file()})]
        hint = f" (found inside '{nested[0].name}' — zip files, not folder)" if nested else ""
        sys.exit(f"Missing: {sorted(missing)}{hint}")

    requirements = bundle_dir / "requirements.txt"
    if requirements.exists():
        import subprocess
        print("Installing requirements.txt …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])

    sys.path.insert(0, str(bundle_dir))
    spec   = importlib.util.spec_from_file_location("submitted_inference", bundle_dir / "inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model  = module.load_model(str(bundle_dir / model_file))
    return module, model


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_pred(bgr: np.ndarray, x: float, y: float) -> None:
    px, py = int(round(x)), int(round(y))
    h, r, c = PRED_HALF, CORNER_R, PRED_COLOR
    cv2.line(bgr, (px-h+r, py-h), (px+h-r, py-h), c, 2, cv2.LINE_AA)
    cv2.line(bgr, (px+h, py-h+r), (px+h, py+h-r), c, 2, cv2.LINE_AA)
    cv2.line(bgr, (px+h-r, py+h), (px-h+r, py+h), c, 2, cv2.LINE_AA)
    cv2.line(bgr, (px-h, py+h-r), (px-h, py-h+r), c, 2, cv2.LINE_AA)
    cv2.ellipse(bgr, (px-h+r, py-h+r), (r,r), 180,  0, 90, c, 2, cv2.LINE_AA)
    cv2.ellipse(bgr, (px+h-r, py-h+r), (r,r), 270,  0, 90, c, 2, cv2.LINE_AA)
    cv2.ellipse(bgr, (px+h-r, py+h-r), (r,r),   0,  0, 90, c, 2, cv2.LINE_AA)
    cv2.ellipse(bgr, (px-h+r, py+h-r), (r,r),  90,  0, 90, c, 2, cv2.LINE_AA)
    cv2.circle(bgr, (px, py), 2, c, -1, cv2.LINE_AA)


def _draw_gt_t1(bgr: np.ndarray, x: float, y: float) -> None:
    gx, gy = int(round(x)), int(round(y))
    cv2.drawMarker(bgr, (gx, gy), GT_COLOR, cv2.MARKER_CROSS, 40, 2, cv2.LINE_AA)
    cv2.circle(bgr, (gx, gy), 10, GT_COLOR, 2, cv2.LINE_AA)


def _draw_quad(bgr: np.ndarray, pts: np.ndarray, color) -> None:
    cv2.polylines(bgr, [pts.astype(np.int32).reshape(-1, 1, 2)],
                  isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    for px, py in pts.tolist():
        cv2.circle(bgr, (int(round(px)), int(round(py))), 6, color, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Task 1 — keypoint visualization
# ---------------------------------------------------------------------------

TASK1_ID = 0

def _load_gt_t1(scenario_dir: Path, frame_id: str):
    path = scenario_dir / "Numerical" / f"{frame_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    gt = data.get("Ground Truth", {}).get("Task 1")
    if gt is None:
        return None
    return float(gt[0]), float(gt[1])


def visualize_task1(module, model, data_path: Path, out_path: Path, fps: float):
    is_video = data_path.is_file() and data_path.suffix.lower() in VIDEO_SUFFIXES

    win    = "Task 1 — Predictions"
    writer = None

    if is_video:
        cap   = cv2.VideoCapture(str(data_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Source : {data_path.name}  ({total} frames)")
        print(f"Output : {out_path}\n")

        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pred = module.inference(TASK1_ID, None, rgb, model)
            px, py = float(pred["keypoints"][0]), float(pred["keypoints"][1])

            frame = bgr.copy()
            _draw_pred(frame, px, py)

            if writer is None:
                h, w   = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
            cv2.imshow(win, frame)
            cv2.waitKey(1)

            total_str = str(total) if total > 0 else "?"
            if idx % 100 == 0:
                print(f"  [{idx+1}/{total_str}]  pred=({px:.1f},{py:.1f})")
            idx += 1
        cap.release()

    else:
        # Scenario folder
        stereo_root = data_path / "Stereo Left"
        frame_ids   = sorted(p.name for p in stereo_root.iterdir() if p.is_dir())
        total       = len(frame_ids)
        print(f"Source : {data_path.name}  ({total} frames)")
        print(f"Output : {out_path}\n")

        for idx, fid in enumerate(frame_ids):
            # Load OCT volume (mirrors ingestion_keypoints.load_oct_volume)
            bscan_dir = data_path / "iOCT Microscope" / "Bscan" / fid
            oct_vol   = None
            if bscan_dir.is_dir():
                slices = sorted(bscan_dir.glob("*.png"))
                if slices:
                    vol = []
                    for p in slices:
                        with Image.open(p) as img:
                            vol.append(np.array(img.convert("L")))
                    oct_vol = np.stack(vol, axis=0)

            with Image.open(data_path / "Stereo Left" / fid / "microscope.png") as img:
                rgb = np.array(img.convert("RGB"))

            pred   = module.inference(TASK1_ID, oct_vol, rgb, model)
            px, py = float(pred["keypoints"][0]), float(pred["keypoints"][1])
            gt     = _load_gt_t1(data_path, fid)

            bgr   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            frame = bgr.copy()
            if gt:
                _draw_gt_t1(frame, gt[0], gt[1])
                cv2.line(frame,
                         (int(round(gt[0])), int(round(gt[1]))),
                         (int(round(px)),    int(round(py))),
                         LINE_COLOR, 1, cv2.LINE_AA)
            _draw_pred(frame, px, py)

            if writer is None:
                h, w   = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
            cv2.imshow(win, frame)
            cv2.waitKey(1)

            if idx % 200 == 0:
                gt_str = f"  gt=({gt[0]:.1f},{gt[1]:.1f})" if gt else ""
                print(f"  [{idx+1}/{total}] {fid}  pred=({px:.1f},{py:.1f}){gt_str}")

    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\nVideo saved → {out_path}")


# ---------------------------------------------------------------------------
# Task 2 — registration visualization
# Same scenario/timestamp layout as Task 1.
# OCT lives in iOCT Microscope/Volume/<timestamp>/*.png
# Fundus is Stereo Left/<timestamp>/microscope.png
# GT homography is in Numerical/<timestamp>.json → Ground Truth.Task 2
# ---------------------------------------------------------------------------

TASK2_ID     = 1
UNIT_CORNERS = np.array([[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=np.float64)


def _project_corners(H):
    proj = (np.asarray(H, dtype=np.float64) @ UNIT_CORNERS.T).T
    return (proj[:, :2] / proj[:, 2:3]).astype(np.float32)


def _load_oct_volume_t2(scenario_dir: Path, frame_id: str):
    """Mirrors ingestion_registration.load_oct_volume."""
    volume_dir = scenario_dir / "iOCT Microscope" / "Volume" / frame_id
    if not volume_dir.is_dir():
        return None
    slices = sorted(p for p in volume_dir.glob("*.png") if p.stem.isdigit())
    if not slices:
        return None
    vol = []
    for p in slices:
        with Image.open(p) as img:
            vol.append(np.array(img.convert("L")))
    return np.stack(vol, axis=0)


def _load_opmi_t2(scenario_dir: Path, frame_id: str) -> np.ndarray:
    with Image.open(scenario_dir / "Stereo Left" / frame_id / "microscope.png") as img:
        return np.array(img.convert("RGB"))


def _load_gt_H_t2(scenario_dir: Path, frame_id: str):
    path = scenario_dir / "Numerical" / f"{frame_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    gt = data.get("Ground Truth", {}).get("Task 2")
    if gt is None:
        return None
    return np.asarray(gt, dtype=np.float64)


def _enumerate_frames_t2(scenario_dir: Path):
    """Return sorted frame_ids that have both Stereo Left and Numerical entries."""
    numerical_ids = {p.stem for p in (scenario_dir / "Numerical").glob("*.json")}
    stereo_ids    = {p.name for p in (scenario_dir / "Stereo Left").iterdir() if p.is_dir()}
    return sorted(numerical_ids & stereo_ids)


def _build_display_t2(bgr: np.ndarray, H_pred, H_gt):
    out = bgr.copy()
    if H_gt is not None:
        _draw_quad(out, _project_corners(H_gt),   GT_COLOR)
    _draw_quad(out, _project_corners(H_pred), PRED_COLOR)
    return out


def visualize_task2(module, model, data_path: Path, out_dir: Path):
    # Accept either the scenario folder directly or a data root with Task 2/ inside
    if (data_path / "Stereo Left").is_dir():
        scenarios = [data_path]
    elif (data_path / "Task 2").is_dir():
        scenarios = sorted(p for p in (data_path / "Task 2").iterdir() if p.is_dir())
    else:
        scenarios = sorted(p for p in data_path.iterdir() if p.is_dir()
                           and (p / "Stereo Left").is_dir())

    if not scenarios:
        sys.exit(f"No Task 2 scenario folders found under: {data_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all (scenario, frame_id) pairs
    all_frames = []
    for scen in scenarios:
        if not (scen / "Numerical").is_dir() or not (scen / "Stereo Left").is_dir():
            continue
        for fid in _enumerate_frames_t2(scen):
            all_frames.append((scen, fid))

    if not all_frames:
        sys.exit("No usable frames found.")

    print(f"Scenarios : {len(scenarios)}")
    print(f"Frames    : {len(all_frames)}")
    print(f"PNGs      : {out_dir}\n")

    saved_pngs = []
    for idx, (scen, fid) in enumerate(all_frames):
        oct_vol  = _load_oct_volume_t2(scen, fid)
        opmi_rgb = _load_opmi_t2(scen, fid)
        pred     = module.inference(TASK2_ID, oct_vol, opmi_rgb, model)
        H_pred   = np.asarray(pred, dtype=np.float64)
        H_gt     = _load_gt_H_t2(scen, fid)

        if H_gt is not None:
            pts_pred = _project_corners(H_pred)
            pts_gt   = _project_corners(H_gt)
            err = float(np.mean(np.linalg.norm(pts_pred - pts_gt, axis=1)))
            print(f"  [{idx+1}/{len(all_frames)}] {scen.name}/{fid}  corner_err={err:.4f}px")
        else:
            print(f"  [{idx+1}/{len(all_frames)}] {scen.name}/{fid}  (no GT)")

        bgr     = cv2.cvtColor(opmi_rgb, cv2.COLOR_RGB2BGR)
        display = _build_display_t2(bgr, H_pred, H_gt)
        png_path = out_dir / f"{scen.name}_{fid}.png"
        cv2.imwrite(str(png_path), display)
        saved_pngs.append(png_path)

    print(f"\n{len(saved_pngs)} PNGs saved to {out_dir}")
    print("Showing slideshow — press any key to advance, Esc/q to quit\n")

    win = "Task 2 — Registration Predictions"
    for i, png in enumerate(saved_pngs):
        bgr = cv2.imread(str(png))
        if bgr is None:
            continue
        cv2.imshow(win, bgr)
        cv2.setWindowTitle(win, f"{png.name}  [{i+1}/{len(saved_pngs)}]")
        if cv2.waitKey(0) & 0xFF in (27, ord("q")):
            break
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run FIDO inference and visualize results (Task 1: video; Task 2: PNGs + slideshow).")
    parser.add_argument("--task", required=True, type=int, choices=[1, 2])
    parser.add_argument("--bundle", required=True,
                        help="Unzipped submission folder (inference.py + model_*.pth)")
    parser.add_argument("--data", required=True,
                        help=("Task 1: video file OR scenario folder. "
                              "Task 2: scenario folder, data root with 'Task 2/' subfolder, "
                              "or a folder of scenario folders."))
    parser.add_argument("--out", default=None,
                        help=("Task 1: output MP4 path. "
                              "Task 2: output directory for PNGs. "
                              "Default: alongside input."))
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Task 1 video frame rate (default: 25)")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle).expanduser().resolve()
    data_path  = Path(args.data).expanduser().resolve()

    if not bundle_dir.is_dir():
        sys.exit(f"Bundle not found: {bundle_dir}")
    if not data_path.exists():
        sys.exit(f"Data path not found: {data_path}")

    print(f"Task   : {args.task}")
    print(f"Bundle : {bundle_dir}")
    print(f"Data   : {data_path}\n")

    print("Loading submission …")
    module, model = load_submission(bundle_dir, args.task)
    print("Model loaded.\n")

    if args.task == 1:
        if args.out:
            out_path = Path(args.out).expanduser().resolve()
        elif data_path.is_file():
            out_path = data_path.with_stem(data_path.stem + "_predictions").with_suffix(".mp4")
        else:
            out_path = data_path.parent / f"{data_path.name}_predictions.mp4"
        visualize_task1(module, model, data_path, out_path, args.fps)

    else:
        if args.out:
            out_dir = Path(args.out).expanduser().resolve()
        else:
            out_dir = Path(__file__).parent / "task2_visualizations"
        visualize_task2(module, model, data_path, out_dir)


if __name__ == "__main__":
    main()
