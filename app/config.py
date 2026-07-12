
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "dataset")

CONFIDENCE_THRESHOLD = 0.5
ALERT_COOLDOWN = 10

RESTRICTED_ZONE = [
    (100, 100),
    (500, 100),
    (500, 400),
    (100, 400)
]

for directory in [SNAPSHOT_DIR, EVIDENCE_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)
