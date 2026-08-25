import streamlit as st
import base64
import os
import json
from datetime import datetime
import pandas as pd
import time
import numpy as np
from urllib import error, parse, request

# Try to import OpenCV with graceful fallback
try:
    import cv2
    from PIL import Image
    OPENCV_AVAILABLE = True
except Exception as e:
    st.warning(f"OpenCV not available: {e}. Running in limited mode.")
    OPENCV_AVAILABLE = False
    # Create dummy cv2 module to prevent crashes
    import types
    cv2 = types.ModuleType('cv2')
    cv2.VideoCapture = lambda x: None
    cv2.cvtColor = lambda x, y: x
    cv2.imdecode = lambda x, y: None
    cv2.polylines = lambda *args: args[0] if args else None
    cv2.imwrite = lambda *args: None
    
    # Create dummy PIL Image
    Image = types.ModuleType('PIL.Image')
    Image.open = lambda x: None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
LOG_DIR = os.path.join(BASE_DIR, "logs")
ALERTS_DB = os.path.join(BASE_DIR, "alerts.json")
GITHUB_REPOSITORY = "shadow780915551-hash/VIGILANT"
GITHUB_SNAPSHOT_BRANCH = "alert-snapshots"
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
    if not OPENCV_AVAILABLE:
        st.warning("OpenCV not available. Cannot load detector in limited mode.")
        return None
        
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

def _github_token():
    return st.secrets.get("GITHUB_TOKEN", os.environ.get("GITHUB_TOKEN"))

def _github_api(path, method="GET", payload=None):
    token = _github_token()
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not configured")

    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    api_request = request.Request(
        f"https://api.github.com/{path}",
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with request.urlopen(api_request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))

def _ensure_snapshot_branch():
    try:
        _github_api(f"repos/{GITHUB_REPOSITORY}/git/ref/heads/{GITHUB_SNAPSHOT_BRANCH}")
    except error.HTTPError as exc:
        if exc.code != 404:
            raise
        main_ref = _github_api(f"repos/{GITHUB_REPOSITORY}/git/ref/heads/master")
        try:
            _github_api(
                f"repos/{GITHUB_REPOSITORY}/git/refs",
                method="POST",
                payload={
                    "ref": f"refs/heads/{GITHUB_SNAPSHOT_BRANCH}",
                    "sha": main_ref["object"]["sha"],
                },
            )
        except error.HTTPError as create_error:
            if create_error.code != 422:  # Another alert may have created it first.
                raise

def _upload_snapshot_to_github(filepath, filename):
    _ensure_snapshot_branch()
    with open(filepath, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("ascii")

    _github_api(
        f"repos/{GITHUB_REPOSITORY}/contents/{parse.quote(f'snapshots/{filename}')}",
        method="PUT",
        payload={
            "message": f"Store alert snapshot {filename}",
            "content": encoded_image,
            "branch": GITHUB_SNAPSHOT_BRANCH,
        },
    )
    return (
        f"https://raw.githubusercontent.com/{GITHUB_REPOSITORY}/"
        f"{GITHUB_SNAPSHOT_BRANCH}/snapshots/{filename}"
    )

def save_snapshot(frame):
    if not OPENCV_AVAILABLE:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alert_{timestamp}.jpg"
    filepath = os.path.join(SNAPSHOT_DIR, filename)
    if not cv2.imwrite(filepath, frame):
        st.error("Could not save the alert snapshot.")
        return None

    try:
        github_url = _upload_snapshot_to_github(filepath, filename)
        os.remove(filepath)
        return github_url
    except (RuntimeError, error.HTTPError, error.URLError) as exc:
        st.error(f"Could not upload the alert snapshot to GitHub: {exc}")
        return None

def process_frame(frame, detector, confidence, cooldown):
    if not OPENCV_AVAILABLE or detector is None:
        return frame
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

if not OPENCV_AVAILABLE:
    st.error("🚫 LIMITED MODE - OpenCV not available on this system")
    st.info("💡 Upload functionality works, but camera and detection are disabled")

st.title("🔒 VIGILANT - AI Surveillance System")
st.markdown("---")

with st.sidebar:
    st.header("Controls")
    
    if not OPENCV_AVAILABLE:
        st.error("⚠️ Limited Mode - OpenCV not available")
        st.info("📁 Upload Media Only")
        mode = "📁 Upload Media"
    else:
        mode = st.radio("Input Mode", ["📷 Live Camera", "📁 Upload Media"], index=1)
    
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
    
    if not OPENCV_AVAILABLE:
        st.error("🚫 OpenCV not available - Limited Mode")
        st.info("Upload functionality available, but camera and detection are disabled.")
        
        uploaded = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
        if uploaded is not None:
            st.success("✅ File uploaded successfully!")
            if uploaded.type.startswith("image"):
                st.image(uploaded, caption="Uploaded Image (Limited Mode - No Processing)")
            else:
                st.video(uploaded)
            st.info("📝 Full processing requires OpenCV installation.")
        else:
            st.info("👆 Upload any image or video to test the interface.")
    else:
        detector = load_detector()
        if detector is None:
            st.error("Failed to load YOLO model. Check model path or internet connection.")
        else:
            if mode == "📷 Live Camera":
                st.caption("Use your browser camera to capture a frame for surveillance analysis.")
                camera_image = st.camera_input("Camera", key="browser_camera")
                if camera_image is not None:
                    file_bytes = np.frombuffer(camera_image.getvalue(), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if frame is None:
                        st.error("Could not read the captured camera image. Please try again.")
                    else:
                        annotated = process_frame(frame, detector, confidence, cooldown)
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, channels="RGB", width='stretch', caption="Processed Camera Capture")
                else:
                    st.info("Open the browser camera, allow permission, then capture a frame to analyze it.")
            else:
                uploaded = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
                if uploaded is not None:
                    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                    if uploaded.type.startswith("image"):
                        img = cv2.imdecode(file_bytes, 1)
                        annotated = process_frame(img, detector, confidence, cooldown)
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, channels="RGB", width='stretch', caption="Processed Image")
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
                            frame_placeholder.image(annotated_rgb, channels="RGB", width='stretch')
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
                if snapshot_path:
                    if snapshot_path.startswith(("https://", "http://")):
                        st.image(snapshot_path, caption="Alert Snapshot")
                    elif os.path.exists(snapshot_path):
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
    st.dataframe(df, width='stretch')
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Alerts CSV",
        data=csv,
        file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    st.markdown("### Download Individual Snapshots")
    for idx, alert in enumerate(st.session_state.alerts):
        snapshot_path = alert.get('snapshot_path')
        if not snapshot_path:
            continue
        
        timestamp = alert.get('timestamp', 'N/A')[:19]
        severity = alert.get('severity', 'N/A')
        
        col1, col2 = st.columns([3, 1])
        col1.write(f"📸 {timestamp} - {severity}")
        
        if snapshot_path.startswith(("https://", "http://")):
            try:
                with request.urlopen(snapshot_path, timeout=10) as response:
                    image_data = response.read()
                col2.download_button(
                    label="⬇️",
                    data=image_data,
                    file_name=f"alert_{timestamp.replace(':', '-')}.jpg",
                    mime="image/jpeg",
                    key=f"download_{idx}"
                )
            except Exception:
                col2.write("❌")
        elif os.path.exists(snapshot_path):
            try:
                with open(snapshot_path, 'rb') as f:
                    image_data = f.read()
                col2.download_button(
                    label="⬇️",
                    data=image_data,
                    file_name=f"alert_{timestamp.replace(':', '-')}.jpg",
                    mime="image/jpeg",
                    key=f"download_{idx}"
                )
            except Exception:
                col2.write("❌")
else:
    st.info("No alerts to display")

st.markdown("---")
st.markdown("VIGILANT - AI Surveillance System | Powered by YOLOv8 & Streamlit")
