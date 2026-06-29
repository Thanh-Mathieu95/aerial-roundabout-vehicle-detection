# Roundabout Vehicle Detection — YOLOv11

> Capstone Project · Machine and Deep Learning 03  
> Traffic analysis at roundabouts using aerial (UAV) imagery

---

## Overview

This project builds an automated vehicle detection system for aerial images of roundabouts, using the **YOLOv11** architecture. The system detects vehicles (cars, trucks, buses, motorcycles) from a top-down perspective captured by UAV cameras.

| Part | Description |
|------|-------------|
| `training/` | Full training notebook: EDA → Augmentation → YOLOv11 → Evaluation |
| Web App | FastAPI inference demo — upload an image or video and get detections instantly |

---

## Detection Results

![Detection sample](assets/detect_sample.jpg)
*YOLOv11 detecting vehicles at a roundabout — top-down aerial view*

---

## Training Insights

### EDA Heatmap — Object Center Distribution
![Heatmap](assets/heatmap.png)
*Spatial bias detected: high vehicle density near the left edge (x < 0.3)*

### Training Charts
![Training results](assets/results.png)
*Loss curves, mAP@50 and mAP@50-95 across epochs*

![Confusion Matrix](assets/confusion_matrix.png)
*Per-class confusion — common misclassification: Truck ↔ Bus (top-down view)*

---

## Project Structure

```
├── training/
│   └── capstone_roundabout.ipynb   # Full training notebook
├── assets/                          # Images for this README
├── templates/
│   └── index.html                   # Web UI
├── static/
│   └── results/                     # Generated outputs (not committed)
├── main.py                          # FastAPI server
├── best.pt                          # Trained YOLOv11 weights
├── run.bat                          # Quick start (Windows)
└── requirements.txt
```

---

## Notebook & Dataset

Both the training notebook and dataset are available on Google Drive:

**[Open on Google Drive](https://drive.google.com/drive/folders/1g-m2ngEMykuLrMfbZTnFc4BsxzUUkSWk?usp=sharing)**

> Run `capstone_roundabout.ipynb` on Google Colab (GPU recommended) to reproduce training and generate `best.pt`.

### Dataset structure (after download)
```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Classes: `car`, `truck`, `bus`, `motorcycle` (in original label ID order)

---

## Methodology

### 1. EDA & Data Exploration
- Label distribution statistics per class
- Bounding box **center heatmap** → revealed spatial bias (vehicles concentrated at left edge `x < 0.3`)
- Displayed ≥ 5 annotated sample images with original bounding boxes

### 2. Augmentation Strategy
Geometric transforms chosen based on heatmap findings to handle vehicle multi-directionality at roundabouts:

| Augmentation | Reason |
|---|---|
| `degrees=180` | Vehicles travel in all directions around the roundabout |
| `fliplr=0.5` | Reduce left/right spatial bias |
| `flipud=0.5` | Handle top-down view symmetry |
| `mosaic=1.0` | Increase vehicle density variety |
| `scale=0.5` | Simulate vehicles at different distances |

### 3. Model Training — YOLOv11
- Architecture: **YOLOv11s / YOLOv11m**
- Input resolution: `imgsz=640` (preserve small object features in aerial imagery)
- Hyperparameter optimization with **Optuna**: `lr0`, optimizer (AdamW/SGD), momentum, scale, mosaic

### 4. Evaluation
- Overall: `mAP@50`, `mAP@50-95`
- Per-class mAP comparison: without augmentation vs. with augmentation
- Confusion matrix analysis: Truck ↔ Bus misclassification from top-down view
- IoU error analysis in the left-edge region (`x < 0.3`)

---

## Results

| Metric | Value |
|--------|-------|
| mAP@50 | — |
| mAP@50-95 | — |
| Best epoch | — |

> Fill in actual values after training.

---

## Run the Web Demo

### Requirements
- Python 3.10+

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the server

**Windows** — double-click `run.bat`

or run manually:

```bash
uvicorn main:app --reload
```

Open your browser at **http://127.0.0.1:8000**

### Usage
1. Upload an image (JPG, PNG, WEBP) or video (MP4, AVI, MOV, MKV)
2. Click **Detect**
3. View bounding boxes and confidence scores for each detected vehicle

---

## Tech Stack

- [YOLOv11](https://github.com/ultralytics/ultralytics) — Object Detection
- [FastAPI](https://fastapi.tiangolo.com/) — Web Backend
- [OpenCV](https://opencv.org/) — Image Processing
- [Optuna](https://optuna.org/) — Hyperparameter Tuning
- Google Colab — GPU Training

---

## Author

**Nguyen Duong Cong Thanh**  
Machine and Deep Learning 03 · March 2026
