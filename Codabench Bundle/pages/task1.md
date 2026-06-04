# Task 1: Surgical Instrument Keypoints Spatial Localisation

Detect surgical tool keypoints in fundus images and estimate tool–tissue distance, leveraging both fundus and intraoperative OCT (iOCT) data. Methods must be robust to single-modality dropout at inference time.

## Metric

Primary metric: **AUC of Euclidean distance error** between predicted and ground-truth keypoints, computed up to a specified pixel threshold.

```text
accuracy(t) = fraction of keypoints with error <= t
AUC = mean accuracy(t) over t = 0..threshold
```

Some test cases are evaluated with only one modality available (fundus-only or iOCT-only) to assess robustness.

## Ranking

- Mean AUC across all test cases (full and ablated), higher is better.
- If inference time exceeds the per-case limit, that case receives score **0**.
- Incomplete submissions are disqualified.

## Data

| Split | Videos | Frames |
|---|---|---|
| Training | 10 | ~25,000 |
| Test (hidden) | 5 | ~5,000 |

Fundus: 1024x1024 RGB. iOCT: 512×512 greyscale (2 slices per frame).
