# Task 2: Intraoperative iOCT to Fundus Registration

Estimate the rigid transformation matrix that aligns a volumetric iOCT scan to the corresponding fundus microscope image (3D-to-2D registration). Methods must run in real time.

## Metric

Primary metric: **AUC of corner error** — the Euclidean distance between the predicted 2D projection of a 3D iOCT point (using the estimated transformation) and its ground-truth 2D location.

```text
corner_error(p) = ||predicted_2D(p) - gt_2D(p)||_2
AUC = mean accuracy(t) over t = 0..threshold
```

## Ranking

- Mean AUC across all test cases, higher is better.
- If inference time exceeds the per-case limit, that case receives score **0**.
- Incomplete submissions are disqualified.

## Data

| Split | Snapshots | iOCT Volumes | Fundus Images |
|---|---|---|---|
| Training | 150 | 150 | 150 |
| Test (hidden) | 30 | 30 | 30 |

Fundus: 1024×1024 RGB. iOCT volume: 128x512x512 greyscale (provided as 128 slices of 512×512).
