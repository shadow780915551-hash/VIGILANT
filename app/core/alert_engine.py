
import time
from app.config import ALERT_COOLDOWN
from app.core.logger import Logger
from app.core.snapshot import Snapshot
from app.core.severity import SeverityCalculator

class AlertEngine:
    def __init__(self):
        self.last_alert_time = 0
        self.logger = Logger()
        self.severity_calculator = SeverityCalculator()
    
    def should_alert(self):
        current_time = time.time()
        return current_time - self.last_alert_time > ALERT_COOLDOWN
    
    def trigger(self, frame, confidence, num_detections, time_in_zone=0):
        if self.should_alert():
            severity = self.severity_calculator.calculate(confidence, num_detections, time_in_zone)
            snapshot_path = Snapshot.save_evidence(frame)
            message = f"ALERT: Severity {severity} - Confidence: {confidence:.2f}, Detections: {num_detections}"
            self.logger.log(message)
            self.last_alert_time = time.time()
            return {
                "severity": severity,
                "confidence": confidence,
                "num_detections": num_detections,
                "snapshot_path": snapshot_path,
                "timestamp": time.time()
            }
        return None
