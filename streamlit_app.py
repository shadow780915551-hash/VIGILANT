import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
from PIL import Image
import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
LOG_DIR = os.path.join(BASE_DIR, "logs")
ALERTS_DB = os.path.join(BASE_DIR, "alerts.json")
CONFIDENCE_THRESHOLD = 0.5
ALERT_COOLDOWN = 10
RESTRICTED_ZONE = [
    (100, 100),
    (500, 100),
    (500, 400),
    (100, 400)
]

for directory in [SNAPSHOT_DIR, EVIDENCE_DIR, LOG_DIR, os.path.join(BASE_DIR, "models")]:
    os.makedirs(directory, exist_ok=True)

if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = 0

def load_alerts():
    if os.path.exists(ALERTS_DB):
        with open(ALERTS_DB, 'r') as f:
            return json.load(f)
    return []

def save_alerts(alerts):
    with open(ALERTS_DB, 'w') as f:
        json.dump(alerts, f)

@st.cache_resource
def load_detector():
    if not os.path.exists(MODEL_PATH):
        st.warning(f"Model file not found at: {MODEL_PATH}")
        st.info("Please train the model first or place yolov8n.pt in the models/ folder.")
        try:
            from ultralytics import YOLO
            st.info("Downloading default yolov8n.pt for demo use...")
            return YOLO("yolov8n.pt")
        except Exception as e:
            st.error(f"Could not download default model: {e}")
            return None
    try:
        from ultralytics import YOLO
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def calculate_severity(confidence, num_detections, time_in_zone):
    base_severity = confidence * num_detections
    if time_in_zone > 5:
        base_severity *= 1.5
    if time_in_zone > 10:
        base_severity *= 2.0
    if base_severity < 1.0:
        return "LOW"
    elif base_severity < 2.0:
        return "MEDIUM"
    else:
        return "HIGH"

def save_snapshot(frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alert_{timestamp}.jpg"
    filepath = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(filepath, frame)
    return filepath

def process_frame(frame, detector, confidence, cooldown):
    results = detector(frame, conf=confidence, classes=[0])
    annotated_frame = results[0].plot()
    points = np.array(RESTRICTED_ZONE, np.int32)
    points = points.reshape((-1, 1, 2))
    annotated_frame = cv2.polylines(annotated_frame, [points], True, (0, 0, 255), 2)
    num_detections = len(results[0].boxes)
    if num_detections > 0:
        max_conf = max([box.conf[0].item() for box in results[0].boxes])
        in_zone = False
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            if is_point_in_polygon((center_x, center_y), RESTRICTED_ZONE):
                in_zone = True
                break
        if in_zone:
            current_time = time.time()
            if current_time - st.session_state.last_alert_time > cooldown:
                severity = calculate_severity(max_conf, num_detections, 0)
                snapshot_path = save_snapshot(annotated_frame)
                alert = {
                    "severity": severity,
                    "confidence": float(max_conf),
                    "num_detections": num_detections,
                    "snapshot_path": snapshot_path,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.alerts.append(alert)
                save_alerts(st.session_state.alerts)
                st.session_state.last_alert_time = current_time
                st.warning(f"🚨 ALERT: {severity} severity - {num_detections} persons detected in restricted zone!")
    return annotated_frame

st.set_page_config(page_title="VIGILANT - AI Surveillance System", layout="wide")
st.title("🔒 VIGILANT - AI Surveillance System")
st.markdown("---")

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Input Mode", ["📷 Live Camera", "📁 Upload Media"], index=1)
    if mode == "📷 Live Camera":
        if st.button("▶️ Start Camera", key="start"):
            st.session_state.camera_active = True
        if st.button("⏹️ Stop Camera", key="stop"):
            st.session_state.camera_active = False
    st.markdown("---")
    st.header("Settings")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, CONFIDENCE_THRESHOLD, 0.05)
    cooldown = st.slider("Alert Cooldown (seconds)", 1, 60, ALERT_COOLDOWN)
    st.markdown("---")
    st.header("Statistics")
    st.session_state.alerts = load_alerts()
    total_alerts = len(st.session_state.alerts)
    high_severity = sum(1 for a in st.session_state.alerts if a.get('severity') == 'HIGH')
    medium_severity = sum(1 for a in st.session_state.alerts if a.get('severity') == 'MEDIUM')
    low_severity = sum(1 for a in st.session_state.alerts if a.get('severity') == 'LOW')
    st.metric("Total Alerts", total_alerts)
    st.metric("High Severity", high_severity, delta_color="inverse")
    st.metric("Medium Severity", medium_severity)
    st.metric("Low Severity", low_severity)

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Surveillance Feed")
    detector = load_detector()
    if detector is None:
        st.error("Failed to load YOLO model. Check model path or internet connection.")
    else:
        if mode == "📷 Live Camera":
            if st.session_state.camera_active:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Failed to open camera. No physical camera detected.")
                    st.info("Try 'Upload Media' mode instead for Streamlit Cloud deployment.")
                    st.session_state.camera_active = False
                else:
                    frame_placeholder = st.empty()
                    stop_button = st.button("Stop Camera in Main View")
                    if stop_button:
                        st.session_state.camera_active = False
                        cap.release()
                    else:
                        while st.session_state.camera_active:
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Failed to read frame")
                                break
                            annotated = process_frame(frame, detector, confidence, cooldown)
                            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(annotated_rgb, channels="RGB", use_column_width=True)
                            time.sleep(0.1)
                        cap.release()
            else:
                st.info("Click 'Start Camera' in the sidebar to begin live surveillance")
        else:
            uploaded = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
            if uploaded is not None:
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                if uploaded.type.startswith("image"):
                    img = cv2.imdecode(file_bytes, 1)
                    annotated = process_frame(img, detector, confidence, cooldown)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, channels="RGB", use_column_width=True, caption="Processed Image")
                else:
                    temp_path = os.path.join(EVIDENCE_DIR, "temp_video.mp4")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded.getbuffer())
                    cap = cv2.VideoCapture(temp_path)
                    frame_placeholder = st.empty()
                    stop_vid = st.button("Stop Video Processing")
                    while cap.isOpened() and not stop_vid:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        annotated = process_frame(frame, detector, confidence, cooldown)
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(annotated_rgb, channels="RGB", use_column_width=True)
                        time.sleep(0.03)
                    cap.release()
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                st.info("Upload an image or video file to test the surveillance system.")

with col2:
    st.header("Recent Alerts")
    if st.session_state.alerts:
        recent_alerts = st.session_state.alerts[-10:]
        for alert in reversed(recent_alerts):
            severity = alert.get('severity', 'UNKNOWN')
            timestamp = alert.get('timestamp', 'N/A')
            conf = alert.get('confidence', 0)
            detections = alert.get('num_detections', 0)
            severity_color = {
                'HIGH': '🔴',
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }.get(severity, '⚪')
            with st.expander(f"{severity_color} {severity} - {timestamp[:19]}"):
                st.write(f"Confidence: {conf:.2f}")
                st.write(f"Detections: {detections}")
                snapshot_path = alert.get('snapshot_path')
                if snapshot_path and os.path.exists(snapshot_path):
                    st.image(snapshot_path, caption="Alert Snapshot")
    else:
        st.info("No alerts recorded yet")

st.markdown("---")
st.header("All Alerts History")
if st.session_state.alerts:
    alert_data = []
    for alert in st.session_state.alerts:
        alert_data.append({
            'Timestamp': alert.get('timestamp', 'N/A')[:19],
            'Severity': alert.get('severity', 'N/A'),
            'Confidence': f"{alert.get('confidence', 0):.2f}",
            'Detections': alert.get('num_detections', 0),
            'Snapshot': alert.get('snapshot_path', 'N/A')
        })
    df = pd.DataFrame(alert_data)
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Alerts CSV",
        data=csv,
        file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
else:
    st.info("No alerts to display")

st.markdown("---")
st.markdown("VIGILANT - AI Surveillance System | Powered by YOLOv8 & Streamlit")
