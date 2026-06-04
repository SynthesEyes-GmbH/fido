<img src="https://syntheseyes.com/posts/images/mc-logos.webp" alt="FIDO Challenge Preview" width="50%">


# FIDO Challenge — Overview

The **Fusion for Intelligent Decision-support in Ophthalmology (FIDO) Challenge** consists of two parallel tasks based on a fully synthetic multimodal retinal surgery dataset (fundus microscope + intraoperative OCT).

---


## Registration & Support
 
- In order to get access to the dataset, please fill out this [form](https://forms.gle/jnpyrEV9w5R8qvSU7).
- In case of any issues or support, please contact [FIDO 2026 Email](fido@syntheseyes.com).
- Find out the scoring code and local tesing code [here](https://github.com/SynthesEyes-GmbH/fido-2026).

## Task 1 — Surgical Instrument Keypoints Spatial Localisation

Detect surgical tool keypoints in fundus images and estimate tool–tissue distance, leveraging both fundus and iOCT data. Methods must remain robust when one modality is absent at inference time.

### Submission format

Required files at the root of the submission zip:

```text
inference.py
model.pth
requirements.txt   # optional — do NOT include numpy or torch, they are pre-installed
other files...
```

Required interface in `inference.py`:

```python
def load_model(model_path):
    ...

def inference(oct_data, fundus_image, model):
    ...
```
- `fundus_image`: RGB numpy array `(H, W, 3)` — 512×512
- `oct_data`: greyscale numpy array `(2, 256, 256)` — two B-scans; may be `None` in ablation test cases where iOCT is absent

The function must return:

```python
{
    "keypoint": [x, y],   # float
    "tool_tissue_distance": z   # float
}
```

---

## Task 2 — Intraoperative iOCT to Fundus Registration

Estimate the rigid 3D-to-2D transformation matrix that aligns the volumetric iOCT scan to the corresponding fundus microscope image. Real-time inference is required.

### Submission format

Required files at the root of the submission zip:

```text
inference.py
model.pth
requirements.txt   # optional — do NOT include numpy or torch, they are pre-installed
other files...
```

Required interface in `inference.py`:

```python
def load_model(model_path):
    ...

def inference(oct_data, fundus_image, model):
    ...
```

- `fundus_image`: RGB numpy array `(H, W, 3)` — 512×512
- `oct_data`: greyscale numpy array `(256, 256, 256)` — full iOCT volume; may be `None` in ablation test cases

The function must return a `(3, 3)` numpy array representing the 2D homography matrix (float64):

```python
np.array([
    [h00, h01, h02],
    [h10, h11, h12],
    [h20, h21, h22],
], dtype=np.float64)
```

---

## Testing Locally

A local evaluation script is provided to test your submission before uploading. Clone the challenge repository, register for the challenge to receive the sample test data, then follow the instructions in the README:

**[https://github.com/SynthesEyes-GmbH/fido-2026](https://github.com/SynthesEyes-GmbH/fido-2026)**

