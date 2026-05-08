#!/usr/bin/env python3
"""
Local evaluation runner for FIDO challenge submissions.

Usage:
    python run_local.py --submission <path_to_submission_folder> [--task keypoints|registration|both]

The submission folder must contain at minimum:
    inference.py
    model.pth           (required)
    requirements.txt    (optional)
"""

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# BUNDLE_ROOT    = ... # Codabench Bundle Path
# TEST_DATA_ROOT = ... # Data Path

BUNDLE_ROOT = Path("D:/MICCAI Challenge/fido-2026/Codabench Bundle")
TEST_DATA_ROOT = Path("D:/MICCAI Challenge/fido-2026/Worker/data/Test Data")

TASKS = {
    "keypoints": {
        "ingestion": BUNDLE_ROOT / "ingestion_program" / "ingestion_keypoints.py",
        "scoring":   BUNDLE_ROOT / "scoring_program"   / "scoring_keypoints.py",
        "phase":     "Phase 1",
    },
    "registration": {
        "ingestion": BUNDLE_ROOT / "ingestion_program" / "ingestion_registration.py",
        "scoring":   BUNDLE_ROOT / "scoring_program"   / "scoring_registration.py",
        "phase":     "Phase 2",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch_data_paths(mod, data_dir: Path):
    """Override the hardcoded Docker paths in a loaded ingestion/scoring module."""
    data_dir = Path(data_dir)
    if hasattr(mod, "CHALLENGE_DATA_DIR"):
        mod.CHALLENGE_DATA_DIR = data_dir
    if hasattr(mod, "OCT_ROOT"):
        mod.OCT_ROOT = data_dir / "OCT"
    if hasattr(mod, "OPMI_ROOT"):
        mod.OPMI_ROOT = data_dir / "Opmi"
    if hasattr(mod, "NUMERICAL_ROOT"):
        mod.NUMERICAL_ROOT = data_dir / "Numerical"


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------
def run_task(task_name: str, submission_dir: Path, task_cfg: dict) -> dict:
    data_dir = TEST_DATA_ROOT / task_cfg["phase"]

    if not data_dir.exists():
        print(f"  [ERROR] Test data not found: {data_dir}")
        return {"score": 0.0, "error": f"Test data directory not found: {data_dir}"}

    with tempfile.TemporaryDirectory(prefix=f"fido_{task_name}_") as tmp:
        predictions_dir = Path(tmp) / "predictions"
        scores_dir      = Path(tmp) / "scores"
        predictions_dir.mkdir()
        scores_dir.mkdir()

        # ---- Ingestion ----
        print(f"\n{'='*60}")
        print(f"  INGESTION  ({task_name})")
        print(f"{'='*60}")
        ing = load_module(task_cfg["ingestion"], f"ingestion_{task_name}")
        patch_data_paths(ing, data_dir)

        saved_argv = sys.argv[:]
        sys.argv = ["ingestion.py", str(predictions_dir), str(submission_dir)]
        try:
            ing.main()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        finally:
            sys.argv = saved_argv

        # Verify output
        pred_file = predictions_dir / "predictions.json"
        if not pred_file.exists():
            print("  [ERROR] ingestion did not produce predictions.json")
            return {"score": 0.0, "error": "No predictions.json produced by ingestion"}

        with pred_file.open() as fh:
            preds = json.load(fh)
        print(f"\n  Predictions written for {len(preds)} case(s): {sorted(preds.keys())}")

        # ---- Scoring ----
        print(f"\n{'='*60}")
        print(f"  SCORING  ({task_name})")
        print(f"{'='*60}")
        scr = load_module(task_cfg["scoring"], f"scoring_{task_name}")
        patch_data_paths(scr, data_dir)

        sys.argv = ["scoring.py", str(predictions_dir), str(scores_dir)]
        try:
            scr.main()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        finally:
            sys.argv = saved_argv

        with (scores_dir / "scores.json").open() as fh:
            scores = json.load(fh)

        return scores


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run FIDO challenge evaluation locally")
    parser.add_argument(
        "--submission", required=True,
        help="Path to the submission folder (must contain inference.py and model.pth)",
    )
    parser.add_argument(
        "--task", choices=["keypoints", "registration", "both"], default="both",
        help="Which task to evaluate (default: both)",
    )
    args = parser.parse_args()

    submission_dir = Path(args.submission).resolve()
    if not submission_dir.is_dir():
        print(f"ERROR: submission folder not found: {submission_dir}")
        sys.exit(1)

    # Hard-error on missing required files
    errors = []
    if not (submission_dir / "inference.py").exists():
        errors.append("inference.py is missing")
    if not (submission_dir / "model.pth").exists():
        errors.append("model.pth is missing")
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        sys.exit(1)

    tasks_to_run = ["keypoints", "registration"] if args.task == "both" else [args.task]

    all_scores = {}
    for task_name in tasks_to_run:
        try:
            scores = run_task(task_name, submission_dir, TASKS[task_name])
        except Exception as exc:
            scores = {"score": 0.0, "error": str(exc)}
        all_scores[task_name] = scores

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for task_name, scores in all_scores.items():
        error = scores.get("error")
        if error:
            print(f"  {task_name:>14}:  FAILED  — {error}")
        else:
            print(f"  {task_name:>14}:  score = {scores.get('score'):.6f}"
                  f"  ({scores.get('num_cases', '?')} cases)"
                  f"  cuda={bool(scores.get('cuda_available'))}")


if __name__ == "__main__":
    main()
