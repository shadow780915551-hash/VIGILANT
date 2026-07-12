
import os
from datetime import datetime
from app.config import LOG_DIR

class Logger:
    def __init__(self):
        self.log_file = os.path.join(LOG_DIR, "alerts.log")
    
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        with open(self.log_file, "a") as f:
            f.write(log_message)
