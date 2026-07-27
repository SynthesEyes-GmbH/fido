import importlib.util
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


import logging


TASK_ID = 0

CHALLENGE_DATA_DIR = Path("/app/data/comp_data/Test Data") / "Task 1"
REQUIRED_FILES = {"inference.py", "model_0.pth"}
PER_CASE_TIME_LIMIT_SECONDS = 20
RANDOM_SEED = 2026
NUM_EVAL_CASES = 100
OCT_DROPOUT_RATE = 0.1


def install_requirements(submission_dir):
    requirements_path = Path(submission_dir) / "requirements.txt"
    if not requirements_path.exists():
        return

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
    )


class MissingSubmissionFiles(Exception):
    """Raised when the submission archive lacks a required file."""


def load_submission_module(submission_dir):

    logging.info(f"{submission_dir} is created for this submission.")

    submission_dir = Path(submission_dir)
    entries = {path.name for path in submission_dir.iterdir() if path.is_file()}
    missing = REQUIRED_FILES - entries
    if missing:
        message = f"Submission is missing required files: {sorted(missing)}"
        nested = sorted(
            path.name
            for path in submission_dir.iterdir()
            if path.is_dir() and not (REQUIRED_FILES - {item.name for item in path.iterdir() if item.is_file()})
        )
        if nested:
            message += (
                f". The required files were found inside the folder '{nested[0]}' instead. "
                "Zip the files themselves, not the folder containing them."
            )
        raise MissingSubmissionFiles(message)

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
        raise FileNotFoundError("No scenario directories found in the test data")

    return scenarios


def frame_ids(scenario_dir):
    numerical_root = scenario_dir / "Numerical"
    stereo_root = scenario_dir / "Stereo Left"
    numerical_ids = {path.stem for path in numerical_root.glob("*.json")}
    stereo_ids = {path.name for path in stereo_root.iterdir() if path.is_dir()}
    ids = sorted(numerical_ids & stereo_ids)
    if not ids:
        raise FileNotFoundError(
            "No complete frames found in the test data. "
            "Expected matching Numerical/*.json and Stereo Left/<frame_id> entries."
        )

    return ids


def selected_cases():
    """Returns a fixed, deterministic subset of (scenario_dir, frame_id) cases.

    The full case list is built across every scenario and sorted before sampling,
    so the selection depends only on the case ids themselves -- not on directory
    iteration order -- and a given dataset always yields the same NUM_EVAL_CASES.
    """
    cases = []
    for scenario_dir in scenario_dirs():
        for frame_id in frame_ids(scenario_dir):
            cases.append((scenario_dir, frame_id))

    cases.sort(key=lambda case: f"{case[0].name}_{case[1]}")

    if len(cases) <= NUM_EVAL_CASES:
        logging.info(f"Evaluating {len(cases)} cases.")
        return cases

    selected = random.Random(RANDOM_SEED).sample(cases, NUM_EVAL_CASES)
    selected.sort(key=lambda case: f"{case[0].name}_{case[1]}")

    logging.info(f"Evaluating a fixed subset of {len(selected)} cases.")

    return selected


def oct_dropout_cases(cases):
    """Returns the set of case ids whose OCT volume is forced to None.

    Sampled from the evaluated cases with a seed derived from RANDOM_SEED, so the
    dropout set is fixed for a given dataset. This is on top of any natural
    dropout, where the Bscan directory is simply absent.
    """
    case_ids = sorted(f"{scenario_dir.name}_{frame_id}" for scenario_dir, frame_id in cases)
    count = round(len(case_ids) * OCT_DROPOUT_RATE)
    dropped = set(random.Random(RANDOM_SEED + 1).sample(case_ids, count))

    logging.info(
        f"{OCT_DROPOUT_RATE:.0%} of cases will be presented without an OCT volume "
        "(oct_volume=None). Your inference() must handle this."
    )

    return dropped


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
        raise ValueError("keypoints must be a 2-element list or tuple")
    x, y = float(value[0]), float(value[1])
    return [x, y]


def validate_prediction(prediction, test_id):
    if not isinstance(prediction, dict):
        raise ValueError(f"Prediction for test {test_id} must be a dictionary")

    if "keypoints" not in prediction:
        raise ValueError(f"Prediction for test {test_id} missing 'keypoints'")
    if "tool_tissue_distance" not in prediction:
        raise ValueError(f"Prediction for test {test_id} missing 'tool_tissue_distance'")

    kps = prediction["keypoints"]
    if not isinstance(kps, list) or len(kps) != 2:
        raise ValueError(f"keypoints for test {test_id} must be a list of length 2")

    validated_kps = validate_point(kps)

    return {
        "keypoints": validated_kps,
        "tool_tissue_distance": float(prediction["tool_tissue_distance"]),
    }


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/output")
    submission_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/app/ingested_program")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        module = load_submission_module(submission_dir)
    except MissingSubmissionFiles as exc:
        # Recorded so the scoring program can show this on the submission page.
        # Only this check is reported; any other failure is left to the logs.
        with (output_dir / "ingestion_error.json").open("w") as handle:
            json.dump({"error": str(exc)}, handle)
        raise

    model = module.load_model(str(submission_dir / f"model_{TASK_ID}.pth"))

    predictions = {}
    case_durations = {}
    timed_out_cases = []
    total_start = time.monotonic()

    cases = selected_cases()
    dropped_oct_cases = oct_dropout_cases(cases)

    for test_id, (scenario_dir, frame_id) in enumerate(cases):
        case_id = f"{scenario_dir.name}_{frame_id}"
        if case_id in dropped_oct_cases:
            oct_volume = None
        else:
            oct_volume = load_oct_volume(scenario_dir, frame_id)
        opmi_image = load_opmi_image(scenario_dir, frame_id)

        case_start = time.monotonic()
        prediction = module.inference(TASK_ID, oct_volume, opmi_image, model)
        elapsed = time.monotonic() - case_start
        case_durations[test_id] = elapsed

        if elapsed > PER_CASE_TIME_LIMIT_SECONDS:
            logging.info(f"Test {test_id} exceeded the per-case time limit and was skipped.")
            timed_out_cases.append(test_id)
            continue

        predictions[case_id] = validate_prediction(prediction, test_id)

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
