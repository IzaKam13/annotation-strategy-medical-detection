# Annotation Strategy Comparison for Medical Object Detection

Comparison of annotation tools (**CVAT vs Label Studio**) and annotation methods (**manual vs SAM-assisted**) for tumor localization in medical images.

---

## Project Overview

This project investigates how different annotation strategies affect object detection performance and annotation efficiency in a medical imaging context.

The focus is not on medical diagnosis, but on **annotation quality, consistency, and impact on model performance**.

---

## Research Questions

* How does annotation tool choice (CVAT vs Label Studio) affect annotation workflow and quality?
* How does annotation method (manual vs SAM-assisted) impact model performance?
* What trade-offs exist between annotation speed and accuracy?

---

## Dataset

### Source

Breast Ultrasound Images Dataset (BUSI)

### Description

* Ultrasound images of breast tissue
* Includes:

  * tumor masks
  * class labels (benign, malignant, normal)

### Usage in this project

* Masks are used as **reference annotations**
* Bounding boxes are derived from masks or drawn manually
* The task is **tumor localization (object detection)**

---

## Disclaimer

This project does not involve medical diagnosis.

Annotations are based on existing labeled datasets and are used to evaluate annotation strategies, not clinical decision-making.

---

## Experimental Setup

### Annotation Variants

| ID | Tool         | Method       | Description                                                    |
| -- | ------------ | ------------ | -------------------------------------------------------------- |
| V0 | None         | Mask -> bbox | Bounding boxes generated automatically from ground truth masks |
| V1 | CVAT         | Manual       | Bounding boxes drawn manually using mask guidance              |
| V2 | Label Studio | Manual       | Same task using Label Studio                                   |
| V3 | CVAT         | SAM-assisted | SAM-generated masks converted to bounding boxes                |
| V4 | Label Studio | SAM-assisted | (optional) SAM + manual refinement                             |

---

### Model

* YOLOv8 (baseline model)
* Same architecture used across all experiments

---

### Pipeline

1. Load dataset (images + masks)
2. Convert masks → bounding boxes (for baseline)
3. Create annotation variants (manual / SAM-assisted)
4. Train YOLOv8 model for each variant
5. Evaluate performance using standard metrics

---

## Project Structure

```text
annotation-strategy-medical-detection/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── annotations/
│   ├── cvat_manual/
│   ├── labelstudio_manual/
│   ├── sam_auto/
│
├── outputs/
│   ├── models/
│   ├── metrics/
│   ├── plots/
│   └── examples/
│
├── src/
│   ├── download_dataset.py
│   ├── inspect_dataset.py
│   ├── check_mask.py
│   ├── mask_to_bbox.py
│   ├── visualize_bbox.py
|   ├── data_processing.py
│   ├── train.py
│   └── evaluate.py
│
├── README.md
└── requirements.txt
```

---

## Results

> *To be filled during experimentation*
Ground truth masks are sometimes conservative and do not fully cover
all visually apparent tumor regions. This may affect bounding box quality
and model performance.

### Quantitative Results

| Method              | mAP@0.5 | Precision | Recall | F1 |
| ------------------- | ------- | --------- | ------ | -- |
| CVAT Manual         | -       | -         | -      | -  |
| Label Studio Manual | -       | -         | -      | -  |
| SAM-assisted        | -       | -         | -      | -  |

---

### Annotation Efficiency

| Method       | Time/Image | Total Time | Notes |
| ------------ | ---------- | ---------- | ----- |
| CVAT Manual  | -          | -          | -     |
| Label Studio | -          | -          | -     |
| SAM-assisted | -          | -          | -     |

---

### Qualitative Results

> Example predictions and annotation differences will be added here

---

## Dataset Processing

* Masks are converted to bounding boxes using a custom script
* YOLO format labels are generated automatically
* Dataset is split into train/val/test sets

---

## Key Insights

> *To be completed after experiments*

* Annotation quality vs speed trade-offs
* Tool usability differences
* Impact of annotation noise on detection performance

---

## Future Improvements

* Extend to segmentation (mask-based models)
* Evaluate additional datasets
* Compare with expert-labeled annotations
* Improve SAM refinement workflow

---

## Why This Project Matters

This project demonstrates:

* Practical experience with annotation tools (CVAT, Label Studio)
* Understanding of data-centric AI workflows
* Ability to design controlled experiments
* Evaluation of annotation strategies in real-world scenarios

---

## Installation

```bash
git clone https://github.com/IzaKam13/annotation-strategy-medical-detection
cd annotation-strategy-medical-detection
pip install -r requirements.txt
```

---

## Progress

### Completed
- Dataset inspection
- Mask-to-bounding-box conversion
