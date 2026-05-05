import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


CHALLENGE_DATA_DIR = Path("/app/data/comp_data/Test Data") / "Phase 2"

OCT_ROOT = CHALLENGE_DATA_DIR / "OCT"
OPMI_ROOT = CHALLENGE_DATA_DIR / "Opmi"
NUMERICAL_ROOT = CHALLENGE_DATA_DIR / "Numerical"
REQUIRED_FILES = {"inference.py", "model.pth"}


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
        raise AttributeError("inference.py must define inference(oct_volume, fundus_image, model)")
    return module


def case_ids():
    oct_ids = {path.name for path in OCT_ROOT.iterdir() if path.is_dir()}
    opmi_ids = {path.name for path in OPMI_ROOT.iterdir() if path.is_dir()}
    numerical_ids = {path.name for path in NUMERICAL_ROOT.iterdir() if path.is_dir()}
    ids = sorted(oct_ids & opmi_ids & numerical_ids)
    if not ids:
        raise FileNotFoundError(
            f"No complete case ids found under {CHALLENGE_DATA_DIR}. "
            "Expected OCT, Opmi, and Numerical subdirectories."
        )
    return ids


def load_oct_volume(case_id):
    slices = sorted((OCT_ROOT / case_id).glob("*.png"))
    if not slices:
        raise FileNotFoundError(f"No OCT slices found for case {case_id}")
    volume = []
    for path in slices:
        with Image.open(path) as image:
            volume.append(np.array(image.convert("L")))
    return np.stack(volume, axis=0)


def load_opmi_image(case_id):
    image_path = OPMI_ROOT / case_id / "image.png"
    with Image.open(image_path) as image:
        return np.array(image.convert("RGB"))


def validate_prediction(prediction, case_id):
    arr = np.asarray(prediction, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(
            f"Prediction for {case_id} must be a (3, 3) homography matrix, got shape {arr.shape}"
        )
    return arr.tolist()


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/output")
    submission_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/app/ingested_program")
    output_dir.mkdir(parents=True, exist_ok=True)

    module = load_submission_module(submission_dir)
    model = module.load_model(str(submission_dir / "model.pth"))

    predictions = {}
    for case_id in case_ids():
        oct_volume = load_oct_volume(case_id)
        opmi_image = load_opmi_image(case_id)
        prediction = module.inference(oct_volume, opmi_image, model)
        predictions[case_id] = validate_prediction(prediction, case_id)

    with (output_dir / "predictions.json").open("w") as handle:
        json.dump(predictions, handle)


if __name__ == "__main__":
    main()
