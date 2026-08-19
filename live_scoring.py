#!/usr/bin/env python3
"""
Live pipeline scoring for FIDO submissions.

Run without arguments — it will prompt for all paths interactively:

    python live_scoring.py

For each case it:
  - Runs inference (reusing the existing ingestion modules verbatim)
  - Scores each case (reusing the existing scoring modules verbatim)
  - Writes an annotated PNG to <bundle>/live_scoring/task_<N>/
  - Prints per-case error and AUC contribution to the terminal
  - Writes per_case_results.json and prints the overall AUC summary
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image



# ---------------------------------------------------------------------------
# Module loader (mirrors run_local.py)
# ---------------------------------------------------------------------------

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch_data_dir(mod, data_dir: Path):
    if hasattr(mod, "CHALLENGE_DATA_DIR"):
        mod.CHALLENGE_DATA_DIR = Path(data_dir)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        val = ""
    return val or default


def ask_task() -> str:
    while True:
        val = ask("Task to score (1 / 2 / both)", "both").lower()
        if val in ("1", "2", "both"):
            return val
        print("  Please enter 1, 2, or both.")


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _put_text(img, text, org, scale=0.65, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,        thickness,     cv2.LINE_AA)


def t1_draw(opmi_rgb, pred_x, pred_y, gt_x, gt_y, case_id,
            kp_err, dist_err, kp_auc, dist_auc, case_auc):
    bgr = cv2.cvtColor(opmi_rgb, cv2.COLOR_RGB2BGR)

    gx, gy = int(round(gt_x)), int(round(gt_y))
    cv2.drawMarker(bgr, (gx, gy), (0, 200, 0), cv2.MARKER_CROSS, 40, 2, cv2.LINE_AA)
    cv2.circle(bgr,    (gx, gy), 10, (0, 200, 0), 2, cv2.LINE_AA)
    _put_text(bgr, f"GT ({gt_x:.1f}, {gt_y:.1f})", (gx + 14, gy - 14), color=(0, 200, 0))

    px, py = int(round(pred_x)), int(round(pred_y))
    cv2.drawMarker(bgr, (px, py), (0, 0, 220), cv2.MARKER_CROSS, 40, 2, cv2.LINE_AA)
    cv2.circle(bgr,    (px, py), 10, (0, 0, 220), 2, cv2.LINE_AA)
    _put_text(bgr, f"Pred ({pred_x:.1f}, {pred_y:.1f})", (px + 14, py - 14), color=(0, 0, 220))

    cv2.line(bgr, (gx, gy), (px, py), (200, 200, 0), 1, cv2.LINE_AA)

    for i, line in enumerate([
        case_id,
        f"KP error: {kp_err:.2f} px   KP AUC: {kp_auc:.4f}",
        f"Dist error: {dist_err:.2f} px   Dist AUC: {dist_auc:.4f}",
        f"Case final AUC: {case_auc:.4f}",
    ]):
        _put_text(bgr, line, (10, 28 + i * 28))

    return bgr


UNIT_CORNERS = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64)


def _project_corners(H):
    projected = (np.asarray(H, dtype=np.float64) @ UNIT_CORNERS.T).T
    w = projected[:, 2:3]
    return (projected[:, :2] / np.where(np.abs(w) < 1e-9, 1e-9, w)).astype(np.float32)


def _draw_quad(img, pts, color, label):
    cv2.polylines(img, [pts.astype(np.int32).reshape(-1, 1, 2)],
                  isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    for i, (px, py) in enumerate(pts.tolist()):
        cv2.circle(img, (int(round(px)), int(round(py))), 6, color, -1, cv2.LINE_AA)
        _put_text(img, f"{label}{i}", (int(round(px)) + 8, int(round(py)) - 8),
                  scale=0.45, color=color, thickness=1)


def t2_draw(opmi_rgb, H_pred, H_gt, case_id, corner_err, case_auc):
    bgr = cv2.cvtColor(opmi_rgb, cv2.COLOR_RGB2BGR)
    _draw_quad(bgr, _project_corners(H_gt),   (0, 200, 0), "GT")
    _draw_quad(bgr, _project_corners(H_pred), (0, 0, 220), "Pr")
    for i, line in enumerate([
        case_id,
        "Green = GT   |   Red = Prediction",
        f"Corner error: {corner_err:.2f} px   AUC: {case_auc:.4f}",
    ]):
        _put_text(bgr, line, (10, 28 + i * 28))
    return bgr


def load_opmi(scenario_dir: Path, frame_id: str) -> np.ndarray:
    path = scenario_dir / "Stereo Left" / frame_id / "microscope.png"
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


# ---------------------------------------------------------------------------
# Task 1
# ---------------------------------------------------------------------------

def run_task1(bundle_dir: Path, data_dir: Path, ingestion_dir: Path, scoring_dir: Path):
    print("\n" + "=" * 60)
    print("  TASK 1 — Keypoint Localisation")
    print("=" * 60)

    ing = load_module(ingestion_dir / "ingestion_keypoints.py", "ing_kp")
    scr = load_module(scoring_dir   / "scoring_keypoints.py",   "scr_kp")
    patch_data_dir(ing, data_dir)
    patch_data_dir(scr, data_dir)

    # Load submission
    sys.path.insert(0, str(bundle_dir))
    module = ing.load_submission_module(str(bundle_dir))
    model  = module.load_model(str(bundle_dir / "model_0.pth"))

    cases       = ing.selected_cases()
    dropout_ids = ing.oct_dropout_cases(cases)

    out_dir = bundle_dir / "live_scoring" / "task_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    kp_errors   = []
    dist_errors = []
    per_case    = []

    for idx, (scenario_dir, frame_id) in enumerate(cases):
        case_id    = f"{scenario_dir.name}_{frame_id}"
        oct_volume = None if case_id in dropout_ids else ing.load_oct_volume(scenario_dir, frame_id)
        opmi_image = ing.load_opmi_image(scenario_dir, frame_id)

        pred      = module.inference(ing.TASK_ID, oct_volume, opmi_image, model)
        pred      = ing.validate_prediction(pred, idx)
        pred_x, pred_y = pred["keypoints"]
        pred_dist = pred["tool_tissue_distance"]

        ref_kp, ref_dist = scr.load_reference(case_id)
        gt_x, gt_y = ref_kp

        import math
        kp_err   = math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
        dist_err = abs(pred_dist - ref_dist)
        kp_errors.append(kp_err)
        dist_errors.append(dist_err)

        c_kp_auc   = scr.auc_from_errors([kp_err],   scr.MAX_THRESHOLD_PX)
        c_dist_auc = scr.auc_from_errors([dist_err],  scr.MAX_THRESHOLD_DIST)
        c_auc      = scr.KEYPOINT_WEIGHT * c_kp_auc + scr.DISTANCE_WEIGHT * c_dist_auc

        print(f"  [{idx + 1:>3}/{len(cases)}] {case_id}")
        print(f"           KP error: {kp_err:.2f} px   KP AUC: {c_kp_auc:.4f}")
        print(f"         Dist error: {dist_err:.2f} px   Dist AUC: {c_dist_auc:.4f}")
        print(f"         Case AUC:   {c_auc:.4f}")

        opmi_rgb = load_opmi(scenario_dir, frame_id)
        frame = t1_draw(opmi_rgb, pred_x, pred_y, gt_x, gt_y, case_id,
                        kp_err, dist_err, c_kp_auc, c_dist_auc, c_auc)
        cv2.imwrite(str(out_dir / f"{case_id}.png"), frame)

        per_case.append({
            "case_id":       case_id,
            "kp_error_px":   round(kp_err, 4),
            "dist_error_px": round(dist_err, 4),
            "kp_auc":        round(c_kp_auc, 6),
            "dist_auc":      round(c_dist_auc, 6),
            "case_auc":      round(c_auc, 6),
            "oct_dropout":   case_id in dropout_ids,
        })

    kp_auc    = scr.auc_from_errors(kp_errors,   scr.MAX_THRESHOLD_PX)
    dist_auc  = scr.auc_from_errors(dist_errors, scr.MAX_THRESHOLD_DIST)
    final_auc = scr.KEYPOINT_WEIGHT * kp_auc + scr.DISTANCE_WEIGHT * dist_auc

    summary = {
        "task": 1,
        "num_cases":    len(cases),
        "keypoint_auc": round(kp_auc, 6),
        "distance_auc": round(dist_auc, 6),
        "final_auc":    round(final_auc, 6),
        "per_case":     per_case,
    }
    results_path = out_dir / "per_case_results.json"
    with results_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Keypoint AUC : {kp_auc:.6f}")
    print(f"  Distance AUC : {dist_auc:.6f}")
    print(f"  Final AUC    : {final_auc:.6f}  ({scr.KEYPOINT_WEIGHT}×KP + {scr.DISTANCE_WEIGHT}×Dist)")
    print(f"\n  Annotated PNGs → {out_dir}")
    print(f"  Results JSON   → {results_path}")
    return summary


# ---------------------------------------------------------------------------
# Task 2
# ---------------------------------------------------------------------------

def run_task2(bundle_dir: Path, data_dir: Path, ingestion_dir: Path, scoring_dir: Path):
    print("\n" + "=" * 60)
    print("  TASK 2 — iOCT-to-Fundus Registration")
    print("=" * 60)

    ing = load_module(ingestion_dir / "ingestion_registration.py", "ing_reg")
    scr = load_module(scoring_dir   / "scoring_registration.py",   "scr_reg")
    patch_data_dir(ing, data_dir)
    patch_data_dir(scr, data_dir)

    sys.path.insert(0, str(bundle_dir))
    module = ing.load_submission_module(str(bundle_dir))
    model  = module.load_model(str(bundle_dir / "model_1.pth"))

    cases = ing.selected_cases()

    out_dir = bundle_dir / "live_scoring" / "task_2"
    out_dir.mkdir(parents=True, exist_ok=True)

    corner_errors = []
    per_case      = []

    for idx, (scenario_dir, vol_id) in enumerate(cases):
        case_id    = f"{scenario_dir.name}_{vol_id}"
        oct_volume = ing.load_oct_volume(scenario_dir, vol_id)
        opmi_image = ing.load_opmi_image(scenario_dir, vol_id)

        pred   = module.inference(ing.TASK_ID, oct_volume, opmi_image, model)
        H_pred = np.asarray(ing.validate_prediction(pred, idx), dtype=np.float64)
        H_gt   = scr.load_reference(case_id)

        ref_corners  = scr.project_corners(H_gt)
        pred_corners = scr.project_corners(H_pred)
        err          = scr.corner_error(pred_corners, ref_corners)
        c_auc        = scr.auc_from_errors([err])
        corner_errors.append(err)

        print(f"  [{idx + 1:>3}/{len(cases)}] {case_id}")
        print(f"         Corner error: {err:.2f} px   AUC: {c_auc:.4f}")

        opmi_rgb = load_opmi(scenario_dir, vol_id)
        frame = t2_draw(opmi_rgb, H_pred, H_gt, case_id, err, c_auc)
        cv2.imwrite(str(out_dir / f"{case_id}.png"), frame)

        per_case.append({
            "case_id":         case_id,
            "corner_error_px": round(err, 4),
            "case_auc":        round(c_auc, 6),
        })

    final_auc = scr.auc_from_errors(corner_errors)

    summary = {
        "task": 2,
        "num_cases":            len(cases),
        "final_auc":            round(final_auc, 6),
        "mean_corner_error_px": round(float(np.mean(corner_errors)), 4),
        "per_case":             per_case,
    }
    results_path = out_dir / "per_case_results.json"
    with results_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Corner Error AUC : {final_auc:.6f}")
    print(f"  Mean corner error: {np.mean(corner_errors):.2f} px")
    print(f"\n  Annotated PNGs → {out_dir}")
    print(f"  Results JSON   → {results_path}")
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _resolve(path_str: str, label: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    if not p.is_dir():
        sys.exit(f"{label} not found: {p}")
    return p


def main():
    # Allow all inputs to be passed as CLI args (e.g. from a VS Code launch
    # config) so the script doesn't block on input() in that context.
    parser = argparse.ArgumentParser(description="FIDO live pipeline scoring", add_help=True)
    parser.add_argument("--bundle", metavar="DIR", help="Submission bundle folder (unzipped)")
    parser.add_argument("--task",  choices=["1", "2", "both"], help="Task to score")
    parser.add_argument("--data",  metavar="DIR", help="Data root folder (contains 'Task 1' and 'Task 2' subfolders)")
    args = parser.parse_args()

    print("=" * 60)
    print("  FIDO Live Pipeline Scoring")
    print("=" * 60)

    codabench_dir = Path(__file__).parent / "Codabench Bundle"
    ingestion_dir = codabench_dir / "ingestion_program"
    scoring_dir   = codabench_dir / "scoring_program"
    for d in (ingestion_dir, scoring_dir):
        if not d.is_dir():
            sys.exit(f"Expected subfolder not found: {d}")

    bundle_str = args.bundle or ask("Submission bundle folder (unzipped)")
    bundle_dir = _resolve(bundle_str, "Submission bundle folder")

    task = args.task or ask_task()

    data_root = _resolve(args.data or ask("Data root folder (contains 'Task 1' and 'Task 2' subfolders)"), "Data root")

    data1 = data2 = None
    if task in ("1", "both"):
        data1 = _resolve(str(data_root / "Task 1"), "Task 1 data folder")
    if task in ("2", "both"):
        data2 = _resolve(str(data_root / "Task 2"), "Task 2 data folder")

    if data1 is not None:
        run_task1(bundle_dir, data1, ingestion_dir, scoring_dir)
    if data2 is not None:
        run_task2(bundle_dir, data2, ingestion_dir, scoring_dir)

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
