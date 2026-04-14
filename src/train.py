# ============================================================================
# ▶  IMPORTS
# ============================================================================

from pathlib import Path
import torch
from ultralytics import YOLO

# ============================================================================
# ▶  USER CONFIGURATION — edit these before running
# ============================================================================

DATA_PATH = Path("data/processed/yolo_v0/dataset.yaml")
EPOCHS = 20
IMG_SIZE = 640
BATCH_SIZE = 16

# ============================================================================
# ▶  HELPER FUNCTIONS
# ============================================================================

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_model(data_path, epochs=20, imgsz=640, batch=16):
    
    device = get_device()
    print(f"Using device: {device}")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs",
        name="exp1"
    )

    return results


# ============================================================================
# ▶  MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    train_model(DATA_PATH, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH_SIZE)