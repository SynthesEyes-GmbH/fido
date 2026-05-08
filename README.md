# FIDO Challenge — Local Evaluation

Before submitting to Codabench, you can run the full evaluation pipeline (ingestion + scoring) on your local machine using `Submission/run_local.py`.

## Requirements

- Python 3.8+
- Install the challenge dependencies if not already present:
  ```bash
  pip install numpy pillow
  ```

## Setup

Clone the repository and point `run_local.py` at your local test data by editing the two path variables near the top of `Submission/run_local.py`:

```python
# Submission/run_local.py
BUNDLE_ROOT    = Path(__file__).parent / "Codabench Bundle"   # path to the Codabench Bundle folder
TEST_DATA_ROOT = Path(__file__).parent / "Testing" / "data" / "Test Data"  # path to your local test data
```

Adjust `TEST_DATA_ROOT` to wherever you have the sample test data stored. The sample test data is provided after registering for the challenge — [register here](link).

## Submission Folder

Your submission folder must contain:

| File | Required | Notes |
|---|---|---|
| `inference.py` | Yes | Must define `load_model(model_path)` and `inference(oct_data, fundus_image, model)` |
| `model.pth` | Yes | Your trained model weights |
| `requirements.txt` | No | Any extra dependencies — auto-installed before inference runs. **Do not include `numpy` or `torch`** — these are already available in the evaluation environment. |

`oct_data` shape depends on the task:
- **Task 1 (Keypoints):** `(2, 256, 256)` — two B-scans (greyscale). May be `None` in ablation test cases where iOCT is absent.
- **Task 2 (Registration):** `(256, 256, 256)` — full iOCT volume (greyscale). May be `None` in ablation test cases.

## Running Locally

From the repo root, run:

```bash
# Evaluate both tasks
python Submission/run_local.py --submission path/to/your/submission_folder

# Evaluate Task 1 only (Keypoints)
python Submission/run_local.py --submission path/to/your/submission_folder --task keypoints

# Evaluate Task 2 only (Registration)
python Submission/run_local.py --submission path/to/your/submission_folder --task registration
```

## Expected Output

```
============================================================
  INGESTION  (keypoints)
============================================================
  Predictions written for 5 case(s): ['00856', '01214', '01428', '01829', '02152']

============================================================
  SCORING  (keypoints)
============================================================

============================================================
  SUMMARY
============================================================
      keypoints:  score = 0.712400  (5 cases)  cuda=True
  registration:  score = 0.654300  (5 cases)  cuda=True
```

## Submitting to Codabench

Once your local scores look good, zip your submission folder and upload it on [Codabench](https://www.codabench.org). The zip must contain `inference.py` and `model.pth` at the root (not inside a subfolder).

```powershell
# PowerShell — from inside your submission folder:
Compress-Archive -Path .\* -DestinationPath my_submission.zip -Force
```

```bash
# Linux/macOS — from inside your submission folder:
zip -r my_submission.zip .
```

> **Note:** `run_local.py` will exit with an error if `inference.py` or `model.pth` are missing.