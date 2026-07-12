
import cv2
import numpy as np
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def draw_polygon(frame, points, color=(0, 0, 255), thickness=2):
    points = np.array(points, np.int32)
    points = points.reshape((-1, 1, 2))
    return cv2.polylines(frame, [points], True, color, thickness)

def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0
