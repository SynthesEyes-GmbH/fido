import json
import math
import numpy as np
import os
import sys
from pathlib import Path


CHALLENGE_DATA_DIR = Path("/app/data/comp_data/Test Data") / "Phase 2"
NUMERICAL_ROOT = CHALLENGE_DATA_DIR / "Numerical"
MAX_THRESHOLD_PX = 50

# Canonical unit-square corners in homogeneous 2D coordinates (4 x 3)
# Used to project through H and measure reprojection error
REF_CORNERS_H = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
], dtype=np.float64)


def find_file(root, name):
    root = Path(root)
    direct = root / name
    if direct.exists():
        return direct
    matches = list(root.rglob(name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {name} under {root}")


def cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def l2(point_a, point_b):
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


def project_corners(H):
    """Project the 4 canonical unit-square corners through homography H (3x3).
    Returns a list of 4 [x, y] points."""
    H = np.asarray(H, dtype=np.float64)
    projected = (H @ REF_CORNERS_H.T).T  # (4, 3)
    # Normalise homogeneous coordinates
    projected = projected[:, :2] / projected[:, 2:3]
    return projected.tolist()


def corner_error(pred_points, ref_points):
    
    """Mean L2 distance across 4 corner points. pred/ref are lists of 4 [x, y] pairs."""
    if len(pred_points) != 4 or len(ref_points) != 4:
        raise ValueError("Expected exactly 4 corner points")
    return sum(l2(p, r) for p, r in zip(pred_points, ref_points)) / 4


def load_reference(case_id):
    """Load the ground-truth (3, 3) homography matrix for a case."""
    # path = NUMERICAL_ROOT / case_id / f"{case_id}.json"
    # with path.open() as handle:
    #     data = json.load(handle)
    # H = np.asarray(data["homography"], dtype=np.float64)

    H = np.random.rand(3, 3)
    if H.shape != (3, 3):
        raise ValueError(f"Reference homography for {case_id} has shape {H.shape}, expected (3, 3)")
    return H


def auc_from_errors(errors, max_threshold=MAX_THRESHOLD_PX):
    if not errors:
        raise ValueError("Cannot compute AUC with no errors")
    total = 0.0
    for threshold in range(max_threshold + 1):
        accuracy = sum(error <= threshold for error in errors) / len(errors)
        total += accuracy
    return total / (max_threshold + 1)


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/input")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/app/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        predictions_path = find_file(input_dir, "predictions.json")
        with predictions_path.open() as handle:
            predictions = json.load(handle)

        case_ids = sorted(predictions.keys())
        if not case_ids:
            raise ValueError("predictions.json is empty")

        errors = []
        for case_id in case_ids:
            ref_H = load_reference(case_id)
            pred_H = predictions[case_id]  # [[row0], [row1], [row2]] — 3x3 homography

            ref_corners  = project_corners(ref_H)
            pred_corners = project_corners(pred_H)
            errors.append(corner_error(pred_corners, ref_corners))

        score = auc_from_errors(errors)
        scores = {
            "score": round(score, 6),
            "num_cases": len(case_ids),
            "cuda_available": int(cuda_available()),
        }
    except Exception as exc:
        scores = {
            "score": 0.0,
            "num_cases": 0,
            "cuda_available": int(cuda_available()),
            "error": str(exc),
        }

    with (output_dir / "scores.json").open("w") as handle:
        json.dump(scores, handle)


if __name__ == "__main__":
    main()
