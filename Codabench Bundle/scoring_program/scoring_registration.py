import html
import json
import logging
import math
import numpy as np
import os
import sys
from pathlib import Path


CHALLENGE_DATA_DIR = Path("/app/data/comp_data/Test Data") / "Task 2"
MAX_THRESHOLD_PX = 10

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
    scenario_name, frame_id = case_id.rsplit("_", 1)
    path = CHALLENGE_DATA_DIR / scenario_name / "Numerical" / f"{frame_id}.json"
    with path.open() as handle:
        data = json.load(handle)
    H = np.asarray(data["Ground Truth"]["Task 2"], dtype=np.float64)

    if H.shape != (3, 3):
        raise ValueError(f"Reference homography for {case_id} has shape {H.shape}, expected (3, 3)")
    return H


def load_duration(input_dir):
    """Read the total ingestion wall-clock time written by the ingestion program.

    Falls back to -1 if durations.json is missing (e.g. older ingestion runs)."""
    try:
        durations_path = find_file(input_dir, "durations.json")
    except FileNotFoundError:
        return -1
    with durations_path.open() as handle:
        data = json.load(handle)
    return data.get("total_seconds", -1)


def auc_from_errors(errors, max_threshold=MAX_THRESHOLD_PX):
    if not errors:
        raise ValueError("Cannot compute AUC with no errors")
    total = 0.0
    for threshold in range(max_threshold + 1):
        accuracy = sum(error <= threshold for error in errors) / len(errors)
        total += accuracy
    return total / (max_threshold + 1)


def read_ingestion_error(input_dir):
    """Returns the message ingestion recorded for a missing-files failure, if any."""
    try:
        error_path = find_file(input_dir, "ingestion_error.json")
    except Exception:
        return None
    try:
        with error_path.open() as handle:
            return json.load(handle).get("error")
    except Exception:
        return None


def write_detailed_results(output_dir, scores, ingestion_error=None):
    """Writes detailed_results.html, shown on the Codabench submission page."""
    error = scores.get("error")
    if error:
        detail = ingestion_error or error
        body = (
            '<p class="bad"><strong>This submission did not produce a score.</strong></p>'
            f"<p>{html.escape(str(detail))}</p>"
        )
        if not ingestion_error:
            body += (
                "<p>Check the ingestion logs on this page for the underlying cause. "
                "A crash during inference means no predictions were written, "
                "which scores 0.</p>"
            )
    else:
        rows = "".join(
            f"<tr><td>{html.escape(label)}</td><td>{html.escape(str(scores[key]))}</td></tr>"
            for label, key in (
                ("Final score", "final_score"),
                ("Cases evaluated", "num_cases"),
                ("Total inference time (s)", "duration"),
                ("CUDA available", "cuda_available"),
            )
            if key in scores
        )
        body = f"<table>{rows}</table>"

    document = (
        "<html><head><meta charset='utf-8'><style>"
        "body{font-family:sans-serif;margin:1rem;}"
        "table{border-collapse:collapse;}"
        "td{border:1px solid #ccc;padding:6px 12px;}"
        "tr td:first-child{font-weight:600;}"
        ".bad{color:#b00020;}"
        "</style></head><body>"
        "<h2>Task 2 &mdash; Registration</h2>"
        f"{body}</body></html>"
    )

    (output_dir / "detailed_results.html").write_text(document, encoding="utf-8")


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

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

        logging.info(f"{len(case_ids)} predictions is loaded for the scoring of Task 2")

        errors = []
        for case_id in case_ids:
            ref_H = load_reference(case_id)
            pred_H = predictions[case_id]  # [[row0], [row1], [row2]] — 3x3 homography

            ref_corners  = project_corners(ref_H)
            pred_corners = project_corners(pred_H)
            errors.append(corner_error(pred_corners, ref_corners))

        score = auc_from_errors(errors)

        logging.info(f"Scoring of Task 2 is done: final_score={score}")

        scores = {
            "final_score": round(score, 6),
            "duration": load_duration(input_dir),
            "num_cases": len(case_ids),
            "cuda_available": int(cuda_available()),
        }
    except Exception as exc:
        scores = {
            "final_score": 0.0,
            "duration": -1,
            "num_cases": 0,
            "cuda_available": int(cuda_available()),
            "error": str(exc),
        }

    write_detailed_results(output_dir, scores, read_ingestion_error(input_dir))

    with (output_dir / "scores.json").open("w") as handle:
        json.dump(scores, handle)


if __name__ == "__main__":
    main()
