"""Evaluate a trained YOLO model on the validation split."""

# ============================================================================
# ▶  IMPORTS
# ============================================================================

import torch
from pathlib import Path
from ultralytics import YOLO

# ============================================================================
# ▶  USER CONFIGURATION
# ============================================================================

DATA_PATH = Path("data/processed/yolo_v0/dataset.yaml")
MODEL_PATH = Path("runs/detect/runs/exp1/weights/best.pt")

# ============================================================================
# ▶  HELPER FUNCTIONS
# ============================================================================

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_model():
    device = get_device()
    print(f"Using device: {device}")
    print(f"Dataset exists: {DATA_PATH.exists()}")
    print(f"Model exists: {MODEL_PATH.exists()}")

    model = YOLO(str(MODEL_PATH))
    metrics = model.val(data=str(DATA_PATH), device=device)

    print("\nMain metrics:")
    print(metrics.results_dict)


# ============================================================================
# ▶  MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    evaluate_model()