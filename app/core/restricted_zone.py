import cv2
import numpy as np
from app.config import RESTRICTED_ZONE
from app.utils import is_point_in_polygon

class RestrictedZone:
    def __init__(self, zone=RESTRICTED_ZONE):
        self.zone = zone
    
    def check(self, bbox):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return is_point_in_polygon((center_x, center_y), self.zone)
    
    def draw(self, frame):
        points = np.array(self.zone, np.int32)
        points = points.reshape((-1, 1, 2))
        return cv2.polylines(frame, [points], True, (0, 0, 255), 2)
