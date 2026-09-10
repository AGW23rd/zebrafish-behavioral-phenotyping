# Zebrafish Behavioral Phenotyping

**Automated locomotion tracking, multi-parametric health scoring, and survival prediction of zebrafish larvae using YOLO-based computer vision.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository contains the complete pipeline for:

1. **Real-time multi-object tracking** of zebrafish larvae in multi-well plates using YOLO (v8/v11/v12) with Unscented Kalman Filter (UKF) and multiple association strategies (SORT, ByteTrack, OC-SORT)
2. **Automated behavioral phenotyping** via a 7-metric composite scoring system (velocity variance, acceleration, thigmotaxis, burst frequency, asymmetry index, V_max, A_max)
3. **Survival prediction** — larvae health scores strongly predict 30-day survival outcomes (Cox PH hazard ratio = 407× for low-score vs. high-score, p < 0.005)
4. **Tracker benchmarking** across 8 synthetic motion types (linear, random walk, Lévy flight, Brownian bridge, correlated random walk, Perlin noise, spiral, curve)

## Key Results

| Metric | Value |
|---|---|
| Low-Score Hazard Ratio (30 dpf) | **407×** (p < 0.005) |
| Mid-Score Hazard Ratio (30 dpf) | **6.09×** (p < 0.005) |
| Cox Model Concordance | 0.80 – 0.95 |
| Sample Size | 277 larvae across 3 trials |

## Repository Structure

```
├── yolo_tracker_v2.py        # Core multi-object tracker (SORT/ByteTrack/OC-SORT + UKF)
├── score_larvae1.py          # 7-metric composite health scoring system
├── benchmark_tracker.py      # Tracker benchmarking framework (8 motion types)
├── metric_evaluator.py       # MOT metrics (MOTA, IDF1, HOTA)
├── analyze_wells.py          # Per-well locomotion summary extraction
├── pub_figures.py            # Publication-ready figure generation (600 DPI)
├── survival_analysis.py      # Kaplan-Meier curves + Cox PH models
├── spatial_density.py        # Spatial density maps & thigmotaxis visualization
├── requirements.txt          # Python dependencies
└── README.md
```

## Installation

```bash
git clone https://github.com/AGW23rd/zebrafish-behavioral-phenotyping.git
cd zebrafish-behavioral-phenotyping
pip install -r requirements.txt
```

> **Note:** For real-time tracking, you also need a YOLO model file (e.g., `yolov8n.pt` or a custom-trained `.pt` file). These are not included in this repository due to file size.

## Usage

### 1. Real-Time Tracking

```python
from yolo_tracker_v2 import YOLOTracker

tracker = YOLOTracker(
    model_path="your_model.pt",
    tracker_type="bytetrack",  # or "sort", "ocsort"
    max_age=30,
    min_hits=3
)

# Process video frames
for frame in video_source:
    tracked_objects = tracker.update(frame)
```

### 2. Larvae Health Scoring

```bash
python score_larvae1.py
```

Reads locomotion data from `runs/locomotion_raw_data*.xlsx` and produces a scored output with:
- **Quality Score** (1–10 scale)
- **Precision Grade** (A/B/C/D)
- Individual metrics: V_max, A_max, bout frequency, thigmotaxis index, asymmetry index

### 3. Tracker Benchmarking

```bash
python benchmark_tracker.py
```

Evaluates tracker performance (MOTA, IDF1, HOTA) across synthetic motion scenarios with configurable object counts and motion types.

### 4. Survival Analysis

```bash
python survival_analysis.py
```

Generates Kaplan-Meier survival curves and Cox proportional hazards models stratified by health-score group.

## Scoring System

The composite health score combines 7 behavioral metrics with empirically calibrated weights:

| Metric | Weight | Direction |
|---|---|---|
| Velocity S.D. (rank) | 20% | Higher = healthier |
| Acceleration S.D. (rank) | 15% | Higher = healthier |
| Thigmotaxis Index | 15% | Lower = healthier (inverted) |
| V_max (rank) | 15% | Higher = healthier |
| A_max (rank) | 10% | Higher = healthier |
| Bout Frequency (rank) | 15% | Higher = healthier |
| Asymmetry Index | 10% | Lower = healthier (inverted) |

## Tracker Architecture

The tracking pipeline implements:

- **Unscented Kalman Filter (UKF)** for non-linear state estimation with 8D state vector `[x, y, w, h, vx, vy, vw, vh]`
- **Hungarian algorithm** (scipy `linear_sum_assignment`) for optimal detection-to-track assignment
- **Gating** to reject implausible associations
- **Optional ReID** via appearance embedding for identity recovery after occlusion

## Citation

If you use this code in your research, please cite:

```bibtex
@software{zebrafish_behavioral_phenotyping,
  title  = {Zebrafish Behavioral Phenotyping: Automated Locomotion Tracking and Survival Prediction},
  author = {AGW23rd},
  year   = {2026},
  url    = {https://github.com/AGW23rd/zebrafish-behavioral-phenotyping}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
