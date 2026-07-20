import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

TASK_ID = 0

CHALLENGE_DATA_DIR = Path("/app/data/comp_data/Test Data") / "Task 1"
REQUIRED_FILES = {"inference.py", "model_0.pth"}
PER_CASE_TIME_LIMIT_SECONDS = 20


def install_requirements(submission_dir):
    requirements_path = Path(submission_dir) / "requirements.txt"
    if not requirements_path.exists():
        return

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
    )


def load_submission_module(submission_dir):

    submission_dir = Path(submission_dir)
    entries = {path.name for path in submission_dir.iterdir() if path.is_file()}
    missing = REQUIRED_FILES - entries
    if missing:
        raise ValueError(f"Submission is missing required files: {sorted(missing)}")

    install_requirements(submission_dir)
    sys.path.insert(0, str(submission_dir))
    module_path = submission_dir / "inference.py"
    spec = importlib.util.spec_from_file_location("submitted_inference", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "load_model"):
        raise AttributeError("inference.py must define load_model(model_path)")
    if not hasattr(module, "inference"):
        raise AttributeError("inference.py must define inference(task_id, oct_volume, opmi_image, model)")
    return module


def scenario_dirs():
    scenarios = sorted(path for path in CHALLENGE_DATA_DIR.iterdir() if path.is_dir())
    if not scenarios:
        raise FileNotFoundError(f"No scenario directories found under {CHALLENGE_DATA_DIR}")
    return scenarios


def frame_ids(scenario_dir):
    numerical_root = scenario_dir / "Numerical"
    stereo_root = scenario_dir / "Stereo Left"
    numerical_ids = {path.stem for path in numerical_root.glob("*.json")}
    stereo_ids = {path.name for path in stereo_root.iterdir() if path.is_dir()}
    ids = sorted(numerical_ids & stereo_ids)
    if not ids:
        raise FileNotFoundError(
            f"No complete frames found under {scenario_dir}. "
            "Expected matching Numerical/*.json and Stereo Left/<frame_id> entries."
        )
    return ids


def load_oct_volume(scenario_dir, frame_id):
    bscan_dir = scenario_dir / "iOCT Microscope" / "Bscan" / frame_id
    if not bscan_dir.is_dir():
        return None
    slices = sorted(bscan_dir.glob("*.png"))
    if not slices:
        return None
    volume = []
    for path in slices:
        with Image.open(path) as image:
            volume.append(np.array(image.convert("L")))
    return np.stack(volume, axis=0)


def load_opmi_image(scenario_dir, frame_id):
    image_path = scenario_dir / "Stereo Left" / frame_id / "microscope.png"
    with Image.open(image_path) as image:
        return np.array(image.convert("RGB"))


def validate_point(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list) or len(value) != 2:
        print(value)
        raise ValueError(f"{value} must be a 2-element list or tuple")
    x, y = float(value[0]), float(value[1])
    return [x, y]


def validate_prediction(prediction, case_id):
    if not isinstance(prediction, dict):
        raise ValueError(f"Prediction for {case_id} must be a dictionary")

    if "keypoints" not in prediction:
        raise ValueError(f"Prediction for {case_id} missing 'keypoints'")
    if "tool_tissue_distance" not in prediction:
        raise ValueError(f"Prediction for {case_id} missing 'tool_tissue_distance'")

    kps = prediction["keypoints"]
    if not isinstance(kps, list) or len(kps) != 2:
        raise ValueError(f"keypoints for {case_id} must be a list of length 2")

    validated_kps = validate_point(kps)

    return {
        "keypoints": validated_kps,
        "tool_tissue_distance": float(prediction["tool_tissue_distance"]),
    }


def main():

    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/output")
    submission_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/app/ingested_program")
    output_dir.mkdir(parents=True, exist_ok=True)

    module = load_submission_module(submission_dir)
    model = module.load_model(str(submission_dir / f"model_{TASK_ID}.pth"))

    predictions = {}
    case_durations = {}
    timed_out_cases = []
    total_start = time.monotonic()

    for scenario_dir in scenario_dirs():
        for frame_id in frame_ids(scenario_dir):
            case_id = f"{scenario_dir.name}_{frame_id}"
            oct_volume = load_oct_volume(scenario_dir, frame_id)
            opmi_image = load_opmi_image(scenario_dir, frame_id)

            case_start = time.monotonic()
            prediction = module.inference(TASK_ID, oct_volume, opmi_image, model)
            elapsed = time.monotonic() - case_start
            case_durations[case_id] = elapsed

            if elapsed > PER_CASE_TIME_LIMIT_SECONDS:
                timed_out_cases.append(case_id)
                continue

            predictions[case_id] = validate_prediction(prediction, case_id)

    total_elapsed = time.monotonic() - total_start

    with (output_dir / "predictions.json").open("w") as handle:
        json.dump(predictions, handle)

    with (output_dir / "durations.json").open("w") as handle:
        json.dump(
            {
                "total_seconds": total_elapsed,
                "per_case_seconds": case_durations,
                "timed_out_cases": timed_out_cases,
                "per_case_time_limit_seconds": PER_CASE_TIME_LIMIT_SECONDS,
            },
            handle,
        )


if __name__ == "__main__":
    main()
