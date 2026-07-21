import json
import logging
import math
import sys
from pathlib import Path


CHALLENGE_DATA_DIR = Path("/app/data/comp_data/Test Data") / "Task 1"
MAX_THRESHOLD_PX = 10
MAX_THRESHOLD_DIST = 10
KEYPOINT_WEIGHT = 0.7
DISTANCE_WEIGHT = 0.3

# Ground Truth.Task 1[2] is stored at 10x the true tool-tip-to-retina pixel
# distance on the B-scan image (e.g. stored 613.7916 -> true 61.37916 px).
# tool_tissue_distance predictions are expected in true B-scan pixel units,
# so the stored ground truth is divided by this factor before computing
# error. This keeps distance_auc on the same pixel scale (and threshold) as
# keypoint_auc, so the two are comparable before KEYPOINT_WEIGHT/DISTANCE_WEIGHT
# are applied.
GROUND_TRUTH_DISTANCE_SCALE = 10


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
        return -1
    with durations_path.open() as handle:
        data = json.load(handle)
    return data.get("total_seconds", -1)


def auc_from_errors(errors, max_threshold=MAX_THRESHOLD_PX):
    if not errors:
        raise ValueError("Cannot compute AUC with no errors")
    total = 0.0
    for threshold in range(max_threshold + 1):
        total += sum(e <= threshold for e in errors) / len(errors)
    return total / (max_threshold + 1)


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

        keypoint_auc = auc_from_errors(keypoint_errors, MAX_THRESHOLD_PX)
        distance_auc = auc_from_errors(distance_errors, MAX_THRESHOLD_DIST)
        final_score = KEYPOINT_WEIGHT * keypoint_auc + DISTANCE_WEIGHT * distance_auc

        logging.info(
            f"Scoring of Task 1 is done: keypoint_auc={keypoint_auc}, "
            f"distance_auc={distance_auc}, final_score={final_score}"
        )

        scores = {
            "final_score": round(final_score, 6),
            "keypoint_auc": round(keypoint_auc, 6),
            "distance_auc": round(distance_auc, 6),
            "duration": load_duration(input_dir),
            "num_cases": len(case_ids),
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

    with (output_dir / "scores.json").open("w") as handle:
        json.dump(scores, handle)


if __name__ == "__main__":
    main()
