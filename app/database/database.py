
import json
import os
from app.config import BASE_DIR

DATABASE_FILE = os.path.join(BASE_DIR, "alerts.json")

class JSONDatabase:
    def __init__(self):
        self.alerts = self.load()
    
    def load(self):
        if os.path.exists(DATABASE_FILE):
            with open(DATABASE_FILE, "r") as f:
                return json.load(f)
        return []
    
    def save(self):
        with open(DATABASE_FILE, "w") as f:
            json.dump(self.alerts, f, indent=2)
    
    def add_alert(self, alert):
        self.alerts.append(alert)
        self.save()
    
    def get_all_alerts(self):
        return self.alerts
    
    def get_recent_alerts(self, limit=10):
        return self.alerts[-limit:][::-1]
