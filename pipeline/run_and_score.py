#!/usr/bin/env python3
"""
Run inference + score for Task 1 or Task 2 using the Codabench bundle programs.

Usage:
    python run_and_score.py --task 1 --bundle <submission_folder> --data <data_root>
    python run_and_score.py --task 2 --bundle <submission_folder> --data <data_root>

--bundle   Unzipped submission folder (contains inference.py + model_*.pth).
--data     Root that contains Task 1/ and/or Task 2/ subfolders, exactly as the
           competition expects (e.g. D:/fido/Worker/data/data/comp_data/Test Data).

The ingestion and scoring programs from the Codabench bundle are loaded verbatim;
only the hardcoded Docker paths are patched to point at your local data root.

Outputs printed to the terminal; scores.json and detailed_results.html are written
to a temp folder and then printed/summarised.
"""
import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

BUNDLE_ROOT = Path(__file__).parent.parent / "Codabench Bundle"
INGESTION = {
    1: BUNDLE_ROOT / "ingestion_program" / "ingestion_keypoints.py",
    2: BUNDLE_ROOT / "ingestion_program" / "ingestion_registration.py",
}
SCORING = {
    1: BUNDLE_ROOT / "scoring_program" / "scoring_keypoints.py",
    2: BUNDLE_ROOT / "scoring_program" / "scoring_registration.py",
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patch(mod, data_root: Path, task: int):
    task_dir = data_root / f"Task {task}"
    if hasattr(mod, "CHALLENGE_DATA_DIR"):
        mod.CHALLENGE_DATA_DIR = task_dir
    if hasattr(mod, "REAL_TEST_SET_DIR"):
        mod.REAL_TEST_SET_DIR = data_root / "Real Test Set"


def main():
    parser = argparse.ArgumentParser(
        description="Run Codabench ingestion + scoring locally for Task 1 or 2.")
    parser.add_argument("--task", required=True, type=int, choices=[1, 2])
    parser.add_argument("--bundle", required=True,
                        help="Unzipped submission folder (inference.py + model_*.pth)")
    parser.add_argument("--data", required=True,
                        help="Data root containing 'Task 1/' and/or 'Task 2/' subfolders")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle).expanduser().resolve()
    data_root  = Path(args.data).expanduser().resolve()

    if not bundle_dir.is_dir():
        sys.exit(f"Bundle not found: {bundle_dir}")
    if not data_root.is_dir():
        sys.exit(f"Data root not found: {data_root}")

    print(f"Task   : {args.task}")
    print(f"Bundle : {bundle_dir}")
    print(f"Data   : {data_root}\n")

    with tempfile.TemporaryDirectory(prefix=f"fido_t{args.task}_") as tmp:
        pred_dir  = Path(tmp) / "predictions"
        score_dir = Path(tmp) / "scores"
        pred_dir.mkdir()
        score_dir.mkdir()

        # ---- Ingestion ----
        print("=" * 60)
        print(f"  INGESTION  (Task {args.task})")
        print("=" * 60)
        ing = _load(INGESTION[args.task], f"ingestion_{args.task}")
        _patch(ing, data_root, args.task)

        saved_argv = sys.argv[:]
        sys.argv = ["ingestion.py", str(pred_dir), str(bundle_dir)]
        try:
            ing.main()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        finally:
            sys.argv = saved_argv

        pred_file = pred_dir / "predictions.json"
        if not pred_file.exists():
            sys.exit("Ingestion did not produce predictions.json")

        with pred_file.open() as f:
            preds = json.load(f)
        print(f"\nPredictions: {len(preds)} case(s)\n")

        # ---- Per-case logging for Task 2 ----
        if args.task == 2:
            scr_ref = _load(SCORING[2], "scoring_2_ref")
            _patch(scr_ref, data_root, 2)
            print("Per-case corner errors:")
            for case_id in sorted(preds.keys()):
                try:
                    H_gt   = np.asarray(scr_ref.load_reference(case_id), dtype=np.float64)
                    H_pred = np.asarray(preds[case_id], dtype=np.float64)
                    ref_c  = scr_ref.project_corners(H_gt)
                    pred_c = scr_ref.project_corners(H_pred)
                    err    = scr_ref.corner_error(pred_c, ref_c)
                    print(f"  {case_id:<40s}  {err:.4f} px")
                except Exception as e:
                    print(f"  {case_id:<40s}  ERROR: {e}")
            print()

        # ---- Scoring ----
        print("=" * 60)
        print(f"  SCORING  (Task {args.task})")
        print("=" * 60)
        scr = _load(SCORING[args.task], f"scoring_{args.task}")
        _patch(scr, data_root, args.task)

        sys.argv = ["scoring.py", str(pred_dir), str(score_dir)]
        try:
            scr.main()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        finally:
            sys.argv = saved_argv

        scores_file = score_dir / "scores.json"
        with scores_file.open() as f:
            scores = json.load(f)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    error = scores.get("error")
    if error:
        print(f"  FAILED: {error}")
    else:
        for k, v in scores.items():
            print(f"  {k:<30} {v}")


if __name__ == "__main__":
    main()
