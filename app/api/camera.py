
import cv2
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.core.detector import Detector
from app.core.restricted_zone import RestrictedZone
from app.core.alert_engine import AlertEngine
from app.database.database import JSONDatabase

router = APIRouter()

detector = Detector()
restricted_zone = RestrictedZone()
alert_engine = AlertEngine()
db = JSONDatabase()
camera_active = False
cap = None

def generate_frames():
    global cap
    while True:
        if not camera_active or cap is None or not cap.isOpened():
            break
        success, frame = cap.read()
        if not success:
            break
        
        results = detector.detect(frame)
        annotated_frame = detector.draw_boxes(frame, results)
        annotated_frame = restricted_zone.draw(annotated_frame)
        
        num_detections = len(results.boxes)
        if num_detections > 0:
            max_conf = max([box.conf[0].item() for box in results.boxes])
            in_zone = False
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if restricted_zone.check((x1, y1, x2, y2)):
                    in_zone = True
                    break
            if in_zone:
                alert = alert_engine.trigger(annotated_frame, max_conf, num_detections)
                if alert:
                    db.add_alert(alert)
        
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@router.post("/start")
async def start_camera():
    global cap, camera_active
    if not camera_active:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            camera_active = True
            return {"status": "camera started"}
        return {"status": "failed to open camera"}
    return {"status": "camera already active"}

@router.post("/stop")
async def stop_camera():
    global cap, camera_active
    if camera_active and cap is not None:
        cap.release()
        camera_active = False
        return {"status": "camera stopped"}
    return {"status": "camera not active"}

@router.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
