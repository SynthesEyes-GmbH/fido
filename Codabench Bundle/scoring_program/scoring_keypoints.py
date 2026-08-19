import html
import json
import logging
import math
import sys
from pathlib import Path


CHALLENGE_DATA_DIR = Path("/app/data/comp_data/Test Data") / "Task 1"
MAX_THRESHOLD_PX = 10
MAX_THRESHOLD_DIST = 20
KEYPOINT_WEIGHT = 0.7
DISTANCE_WEIGHT = 0.3

# Ground Truth.Task 1[2] is stored at ~7.8x the true tool-tip-to-retina pixel
# distance on the B-scan image (e.g. stored 613.7916 -> true 78.69 px).
# tool_tissue_distance predictions are expected in true B-scan pixel units,
# so the stored ground truth is divided by this factor before computing
# error. This keeps distance_auc on the same pixel scale (and threshold) as
# keypoint_auc, so the two are comparable before KEYPOINT_WEIGHT/DISTANCE_WEIGHT
# are applied.
GROUND_TRUTH_DISTANCE_SCALE = (4.0 / 512) * 1000


def cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def find_file(root, name):
    root = Path(root)
    direct = root / name
    if direct.exists():
        return direct
    matches = list(root.rglob(name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {name} under {root}")

def load_reference(case_id):
    """Returns (keypoint [x, y], tool_tissue_distance_px) ground truth for a case.

    tool_tissue_distance_px is the true tool-tip-to-retina pixel distance on
    the B-scan image, i.e. the stored value divided by GROUND_TRUTH_DISTANCE_SCALE."""
    scenario_name, frame_id = case_id.rsplit("_", 1)
    path = CHALLENGE_DATA_DIR / scenario_name / "Numerical" / f"{frame_id}.json"
    with path.open() as handle:
        data = json.load(handle)
    ground_truth = data["Ground Truth"]["Task 1"]
    return ground_truth[:2], ground_truth[2] / GROUND_TRUTH_DISTANCE_SCALE


def load_duration(input_dir):
    """Read the total ingestion wall-clock time written by the ingestion program.

    Falls back to -1 if durations.json is missing (e.g. older ingestion runs)."""
    try:
        durations_path = find_file(input_dir, "durations.json")
    except FileNotFoundError:
        return -1, []
    with durations_path.open() as handle:
        data = json.load(handle)
    return data.get("total_seconds", -1), data.get("timed_out_cases", [])


def auc_from_errors(errors, max_threshold=MAX_THRESHOLD_PX):
    if not errors:
        raise ValueError("Cannot compute AUC with no errors")
    total = 0.0
    for threshold in range(max_threshold + 1):
        total += sum(e <= threshold for e in errors) / len(errors)
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
        pct = min(max(float(value) / float(max_val), 0.0), 1.0) * 100
        fill = "#1a7f37" if pct >= 50 else ("#d68910" if pct >= 20 else "#b00020")
        return (
            f'<div style="background:#e9ecef;border-radius:6px;height:14px;width:200px;display:inline-block;vertical-align:middle;">'
            f'<div style="background:{fill};width:{pct:.1f}%;height:100%;border-radius:6px;"></div></div> '
            f'<span style="font-size:0.85em;color:#555;">{pct:.1f}%</span>'
        )

    def row(label, value, bar=None):
        bar_td = f'<td style="padding:8px 12px;border-bottom:1px solid #dee2e6;">{bar}</td>' if bar is not None else '<td style="padding:8px 12px;border-bottom:1px solid #dee2e6;"></td>'
        return (
            f'<tr><td style="padding:8px 12px;border-bottom:1px solid #dee2e6;font-weight:600;">{label}</td>'
            f'<td style="padding:8px 12px;border-bottom:1px solid #dee2e6;">{value}</td>'
            f'{bar_td}</tr>'
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
        final_score  = scores.get("final_score", 0)
        kp_auc       = scores.get("keypoint_auc", 0)
        dist_auc     = scores.get("distance_auc", 0)
        num_cases    = scores.get("num_cases", 0)
        duration     = scores.get("duration", -1)
        cuda         = scores.get("cuda_available", 0)

        cuda_badge = (
            '<span style="background:#d4edda;color:#1a7f37;padding:2px 8px;border-radius:10px;font-size:0.85em;">&#9989; GPU</span>'
            if cuda else
            '<span style="background:#f8d7da;color:#721c24;padding:2px 8px;border-radius:10px;font-size:0.85em;">&#10060; CPU only</span>'
        )

        duration_str = f"{duration:.1f} s" if duration >= 0 else "N/A"
        if duration > 0 and num_cases > 0:
            duration_str += f" &nbsp;&#x2022;&nbsp; {duration / num_cases:.2f} s / case"

        body = (
            # Score hero
            '<div style="background:linear-gradient(135deg,#0d6efd11,#0dcaf011);'
            'border:1px solid #b6d4fe;border-radius:10px;padding:16px 20px;margin-bottom:20px;">'
            '<div style="font-size:0.8em;color:#555;text-transform:uppercase;letter-spacing:.05em;">Final Score (0&ndash;1)</div>'
            f'<div style="font-size:2.4em;font-weight:800;margin:4px 0;">{badge(final_score)}</div>'
            f'<div>{pct_bar(final_score)}</div>'
            '<div style="font-size:0.78em;color:#777;margin-top:6px;">'
            f'Weighted combination: {KEYPOINT_WEIGHT}&times;Keypoint AUC + {DISTANCE_WEIGHT}&times;Distance AUC. Higher is better.</div>'
            '</div>'

            # Sub-score cards
            '<div style="display:flex;gap:12px;margin-bottom:20px;">'

            '<div style="flex:1;border:1px solid #dee2e6;border-radius:8px;padding:12px 16px;">'
            '<div style="font-size:0.75em;color:#555;text-transform:uppercase;letter-spacing:.05em;">Keypoint AUC</div>'
            f'<div style="font-size:1.6em;font-weight:700;margin:4px 0;">{badge(kp_auc)}</div>'
            f'<div>{pct_bar(kp_auc)}</div>'
            f'<div style="font-size:0.75em;color:#888;margin-top:4px;">Weight: {KEYPOINT_WEIGHT}</div>'
            '</div>'

            '<div style="flex:1;border:1px solid #dee2e6;border-radius:8px;padding:12px 16px;">'
            '<div style="font-size:0.75em;color:#555;text-transform:uppercase;letter-spacing:.05em;">Distance AUC</div>'
            f'<div style="font-size:1.6em;font-weight:700;margin:4px 0;">{badge(dist_auc)}</div>'
            f'<div>{pct_bar(dist_auc)}</div>'
            f'<div style="font-size:0.75em;color:#888;margin-top:4px;">Weight: {DISTANCE_WEIGHT}</div>'
            '</div>'
            '</div>'

            # Details table
            '<table style="border-collapse:collapse;width:100%;margin-bottom:16px;">'
            '<thead><tr style="background:#f1f3f5;">'
            '<th style="text-align:left;padding:8px 12px;font-size:0.8em;color:#555;border-bottom:2px solid #dee2e6;">Metric</th>'
            '<th style="text-align:left;padding:8px 12px;font-size:0.8em;color:#555;border-bottom:2px solid #dee2e6;">Value</th>'
            '<th style="text-align:left;padding:8px 12px;font-size:0.8em;color:#555;border-bottom:2px solid #dee2e6;">Visualisation</th>'
            '</tr></thead><tbody>'
            + row("Keypoint AUC",       f"{kp_auc:.6f}",   pct_bar(kp_auc))
            + row("Distance AUC",       f"{dist_auc:.6f}",  pct_bar(dist_auc))
            + row("Final score",        f"{final_score:.6f}", pct_bar(final_score))
            + row("Cases evaluated",    str(num_cases))
            + row("Inference time",     duration_str)
            + f'<tr><td style="padding:8px 12px;font-weight:600;">Hardware</td>'
              f'<td style="padding:8px 12px;" colspan="2">{cuda_badge}</td></tr>'
            + '</tbody></table>'

            # Explainer
            '<details style="border:1px solid #dee2e6;border-radius:6px;padding:10px 14px;">'
            '<summary style="cursor:pointer;font-weight:600;font-size:0.9em;">How is the score computed?</summary>'
            '<p style="margin:10px 0 0;font-size:0.85em;color:#444;">'
            '<strong>Keypoint AUC</strong>: L2 pixel distance between predicted and ground-truth tool-tip keypoint. '
            'AUC at thresholds 0&ndash;10 px. '
            '<strong>Distance AUC</strong>: absolute error of the predicted tool-tissue distance vs ground truth (B-scan pixels). '
            'AUC at thresholds 0&ndash;10 px. '
            f'<strong>Final score</strong> = {KEYPOINT_WEIGHT}&times;Keypoint AUC + {DISTANCE_WEIGHT}&times;Distance AUC. '
            'AUC = 1.0 means all predictions are within every threshold &mdash; a perfect submission.</p>'
            '</details>'
        )

    document = (
        "<html><head><meta charset='utf-8'><style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "margin:1.2rem;color:#212529;font-size:14px;}"
        "h2{margin-top:0;color:#0d6efd;font-size:1.3em;}"
        "</style></head><body>"
        "<h2>Task 1 &mdash; Keypoints &amp; Tool-Tissue Distance</h2>"
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

        logging.info(f"{len(case_ids)} predictions is loaded for the scoring of Task 1")

        keypoint_errors = []
        distance_errors = []
        for case_id in case_ids:
            ref_kp, ref_dist = load_reference(case_id)
            kp = predictions[case_id]["keypoints"]
            keypoint_errors.append(math.sqrt((kp[0] - ref_kp[0]) ** 2 + (kp[1] - ref_kp[1]) ** 2))

            pred_dist = predictions[case_id]["tool_tissue_distance"]
            distance_errors.append(abs(pred_dist - ref_dist))

        # Timed-out cases are penalised with an error beyond the max threshold
        # so the AUC denominator is always the total number of cases, not just
        # the ones that finished within the time limit.
        duration, timed_out_cases = load_duration(input_dir)
        PENALTY_ERROR = max(MAX_THRESHOLD_PX, MAX_THRESHOLD_DIST) + 1  # fails every threshold
        keypoint_errors.extend([PENALTY_ERROR] * len(timed_out_cases))
        distance_errors.extend([PENALTY_ERROR] * len(timed_out_cases))
        total_cases = len(keypoint_errors)

        keypoint_auc = auc_from_errors(keypoint_errors, MAX_THRESHOLD_PX)
        distance_auc = auc_from_errors(distance_errors, MAX_THRESHOLD_DIST)
        final_score = KEYPOINT_WEIGHT * keypoint_auc + DISTANCE_WEIGHT * distance_auc

        logging.info(
            f"Scoring of Task 1 is done: keypoint_auc={keypoint_auc}, "
            f"distance_auc={distance_auc}, final_score={final_score} "
            f"({len(case_ids)} predictions + {len(timed_out_cases)} timed-out penalties)"
        )

        scores = {
            "final_score": round(final_score, 6),
            "task1_score": round(final_score, 6),
            "keypoint_auc": round(keypoint_auc, 6),
            "distance_auc": round(distance_auc, 6),
            "duration": duration,
            "num_cases": total_cases,
            "timed_out": len(timed_out_cases),
            "cuda_available": int(cuda_available()),
        }
    except Exception as exc:
        print(exc)
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
