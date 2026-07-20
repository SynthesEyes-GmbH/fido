# Task 1: Surgical Instrument Keypoints Spatial Localisation

Detect surgical tool keypoints in fundus images and estimate tool–tissue distance, leveraging both fundus and intraoperative OCT (iOCT) data. Methods must be robust to single-modality dropout at inference time.

## Metric

Primary metric: a weighted combination of two AUC scores — **keypoint AUC** (70%) and **tool–tissue distance AUC** (30%) — both computed in pixels, up to the same error threshold.

```text
accuracy(t) = fraction of predictions with error <= t
AUC = mean accuracy(t) over t = 0..threshold

keypoint_auc  = AUC of Euclidean pixel distance between predicted and ground-truth keypoints (threshold: 10 px)
distance_auc  = AUC of absolute error between predicted and ground-truth tool-tip-to-retina distance,
                measured in pixels on the B-scan image (threshold: 10 px)

final_score = 0.7 * keypoint_auc + 0.3 * distance_auc
```

`tool_tissue_distance` predictions must be in the same pixel units as the B-scan image (tool-tip-to-retina pixel distance), not millimetres.

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
