
import cv2
import os
from app.config import SNAPSHOT_DIR, EVIDENCE_DIR
from app.utils import get_timestamp

class Snapshot:
    @staticmethod
    def save_snapshot(frame, prefix="snapshot"):
        filename = f"{prefix}_{get_timestamp()}.jpg"
        filepath = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(filepath, frame)
        return filepath
    
    @staticmethod
    def save_evidence(frame, prefix="evidence"):
        filename = f"{prefix}_{get_timestamp()}.jpg"
        filepath = os.path.join(EVIDENCE_DIR, filename)
        cv2.imwrite(filepath, frame)
        return filepath
