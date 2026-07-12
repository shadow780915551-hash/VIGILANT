
from ultralytics import YOLO
import cv2
from app.config import MODEL_PATH, CONFIDENCE_THRESHOLD

class Detector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
    
    def detect(self, frame):
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, classes=[0])
        return results[0]
    
    def draw_boxes(self, frame, results):
        annotated_frame = results.plot()
        return annotated_frame
