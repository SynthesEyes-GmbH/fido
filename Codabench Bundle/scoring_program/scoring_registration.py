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

    def badge(value, good_thresh=0.5, warn_thresh=0.2):
        """Return a colored badge span for a 0–1 score value."""
        v = float(value)
        if v >= good_thresh:
            color, bg = "#1a7f37", "#d4edda"
        elif v >= warn_thresh:
            color, bg = "#856404", "#fff3cd"
        else:
            color, bg = "#721c24", "#f8d7da"
        return (
            f'<span style="background:{bg};color:{color};padding:2px 10px;'
            f'border-radius:12px;font-weight:700;font-size:1.05em;">{v:.4f}</span>'
        )

    def pct_bar(value, max_val=1.0):
        """Return an inline SVG-style progress bar."""
        pct = min(max(float(value) / float(max_val), 0.0), 1.0) * 100
        if pct >= 50:
            fill = "#1a7f37"
        elif pct >= 20:
            fill = "#d68910"
        else:
            fill = "#b00020"
        return (
            f'<div style="background:#e9ecef;border-radius:6px;height:14px;width:200px;display:inline-block;vertical-align:middle;">'
            f'<div style="background:{fill};width:{pct:.1f}%;height:100%;border-radius:6px;"></div></div> '
            f'<span style="font-size:0.85em;color:#555;">{pct:.1f}%</span>'
        )

    error = scores.get("error")

    if error:
        detail = ingestion_error or error
        body = (
            '<div style="background:#f8d7da;border:1px solid #f5c6cb;border-radius:6px;padding:12px 16px;margin-bottom:12px;">'
            '<p style="margin:0 0 6px;color:#721c24;"><strong>&#10060; This submission did not produce a score.</strong></p>'
            f'<p style="margin:0;color:#721c24;font-family:monospace;font-size:0.9em;">{html.escape(str(detail))}</p>'
            '</div>'
        )
        if not ingestion_error:
            body += (
                '<p style="color:#555;font-size:0.9em;">'
                'Check the ingestion logs on this page for the underlying cause. '
                'A crash during inference means no predictions were written, which scores 0.</p>'
            )
    else:
        final_score = scores.get("final_score", 0)
        num_cases = scores.get("num_cases", 0)
        duration = scores.get("duration", -1)
        cuda = scores.get("cuda_available", 0)
        corner_errs_mean = scores.get("mean_corner_error_px")

        cuda_badge = (
            '<span style="background:#d4edda;color:#1a7f37;padding:2px 8px;border-radius:10px;font-size:0.85em;">&#9989; GPU</span>'
            if cuda else
            '<span style="background:#f8d7da;color:#721c24;padding:2px 8px;border-radius:10px;font-size:0.85em;">&#10060; CPU only</span>'
        )

        duration_str = f"{duration:.1f} s" if duration >= 0 else "N/A"
        if duration > 0 and num_cases > 0:
            per_case = duration / num_cases
            duration_str += f" &nbsp;&#x2022;&nbsp; {per_case:.2f} s / case"

        body = (
            # Score hero
            '<div style="background:linear-gradient(135deg,#0d6efd11,#0dcaf011);'
            'border:1px solid #b6d4fe;border-radius:10px;padding:16px 20px;margin-bottom:20px;">'
            '<div style="font-size:0.8em;color:#555;text-transform:uppercase;letter-spacing:.05em;">AUC Score (0–1)</div>'
            f'<div style="font-size:2.4em;font-weight:800;margin:4px 0;">{badge(final_score)}</div>'
            f'<div>{pct_bar(final_score)}</div>'
            '<div style="font-size:0.78em;color:#777;margin-top:6px;">'
            'Area Under the Corner Error Curve at thresholds 0&ndash;10 px. Higher is better.</div>'
            '</div>'

            # Metrics table
            '<table style="border-collapse:collapse;width:100%;margin-bottom:16px;">'
            '<thead><tr style="background:#f1f3f5;">'
            '<th style="text-align:left;padding:8px 12px;font-size:0.8em;color:#555;border-bottom:2px solid #dee2e6;">Metric</th>'
            '<th style="text-align:left;padding:8px 12px;font-size:0.8em;color:#555;border-bottom:2px solid #dee2e6;">Value</th>'
            '<th style="text-align:left;padding:8px 12px;font-size:0.8em;color:#555;border-bottom:2px solid #dee2e6;">Visualisation</th>'
            '</tr></thead><tbody>'
            f'<tr><td style="padding:8px 12px;border-bottom:1px solid #dee2e6;font-weight:600;">Corner Error AUC</td>'
            f'<td style="padding:8px 12px;border-bottom:1px solid #dee2e6;">{final_score:.6f}</td>'
            f'<td style="padding:8px 12px;border-bottom:1px solid #dee2e6;">{pct_bar(final_score)}</td></tr>'
        )
        if corner_errs_mean is not None:
            body += (
                f'<tr><td style="padding:8px 12px;border-bottom:1px solid #dee2e6;font-weight:600;">Mean corner error</td>'
                f'<td style="padding:8px 12px;border-bottom:1px solid #dee2e6;">{corner_errs_mean:.2f} px</td>'
                f'<td style="padding:8px 12px;border-bottom:1px solid #dee2e6;"></td></tr>'
            )
        body += (
            f'<tr><td style="padding:8px 12px;border-bottom:1px solid #dee2e6;font-weight:600;">Cases evaluated</td>'
            f'<td style="padding:8px 12px;border-bottom:1px solid #dee2e6;" colspan="2">{num_cases}</td></tr>'
            f'<tr><td style="padding:8px 12px;border-bottom:1px solid #dee2e6;font-weight:600;">Inference time</td>'
            f'<td style="padding:8px 12px;border-bottom:1px solid #dee2e6;" colspan="2">{duration_str}</td></tr>'
            f'<tr><td style="padding:8px 12px;font-weight:600;">Hardware</td>'
            f'<td style="padding:8px 12px;" colspan="2">{cuda_badge}</td></tr>'
            '</tbody></table>'

            # Explainer
            '<details style="border:1px solid #dee2e6;border-radius:6px;padding:10px 14px;">'
            '<summary style="cursor:pointer;font-weight:600;font-size:0.9em;">How is the score computed?</summary>'
            '<p style="margin:10px 0 0;font-size:0.85em;color:#444;">'
            'The predicted 3&times;3 homography <em>H</em> is used to project the four corners of the '
            'canonical unit square. The mean L2 distance between predicted and ground-truth corners gives the '
            '<strong>corner error</strong> (in pixels). The '
            '<strong>AUC</strong> is the area under the accuracy-vs-threshold curve at integer thresholds '
            '0&ndash;10 px, normalised to [0, 1]. A perfect prediction gives AUC = 1.0.</p>'
            '</details>'
        )

    document = (
        "<html><head><meta charset='utf-8'><style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "margin:1.2rem;color:#212529;font-size:14px;}"
        "h2{margin-top:0;color:#0d6efd;font-size:1.3em;}"
        "</style></head><body>"
        "<h2>Task 2 &mdash; iOCT&ndash;to&ndash;Fundus Registration</h2>"
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
