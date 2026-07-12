
# VIGILANT - Visual Intelligence for Ground Intrusion & Live Alert Notification Technology

A real-time AI-powered surveillance system that detects intrusions using YOLOv8, generates alerts, saves evidence, and displays everything on a modern dashboard.

## Features

- **Live Webcam Feed**: Real-time video streaming with object detection
- **YOLOv8 Person Detection**: Accurate human detection using state-of-the-art YOLOv8 model
- **Restricted Zone Detection**: Define custom restricted zones and get alerts when someone enters
- **Threat Severity Levels**: LOW, MEDIUM, HIGH based on confidence, number of detections, and time in zone
- **Automatic Snapshot Capture**: Save evidence automatically when alerts are triggered
- **Modern Dashboard**: Dark theme with glassmorphism design, real-time stats, and alert history
- **Cooldown Logic**: Avoid duplicate alerts within configurable time window

## Tech Stack

- **Backend**: Python, FastAPI
- **Computer Vision**: YOLOv8, OpenCV
- **Frontend**: HTML, CSS, JavaScript, Jinja2 Templates
- **Database**: JSON (with architecture ready for MongoDB)

## Project Structure

```
VIGILANT/
├── app/
│   ├── api/              # API endpoints (camera, alerts, dashboard)
│   ├── core/             # Core functionality (detector, alert engine, etc.)
│   ├── database/         # Database handling
│   ├── static/           # CSS, JS, images
│   ├── templates/        # HTML templates
│   ├── config.py         # Configuration settings
│   └── utils.py          # Utility functions
├── dataset/              # Dataset directory (COCO format)
├── models/               # Trained YOLOv8 models
├── ML/                   # ML training and evaluation scripts
├── logs/                 # Alert logs
├── snapshots/            # Captured snapshots
├── evidence/             # Evidence from alerts
├── requirements.txt      # Python dependencies
├── run.py                # Run the application
└── README.md             # This file
```

## Installation

1. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Model (if not training your own):**
   The system will automatically download `yolov8n.pt` on first run if it's not present in the `models/` directory.

## Usage

### Training Your Own Model (Optional)

To train YOLOv8 models on COCO dataset with 25 and 35 epochs:

```bash
python ML/train_and_evaluate.py
```

This will:
1. Convert COCO annotations to YOLO format
2. Split dataset into train/validation sets
3. Train models for 25 and 35 epochs
4. Save trained models to `models/` directory
5. Generate evaluation metrics and ROC curves in `ML/` directory

### Running the Application

Start the FastAPI server:

```bash
python run.py
```

Then open your browser and navigate to:
- **Dashboard**: `http://localhost:8000/`
- **Alerts History**: `http://localhost:8000/alerts`

### Using the Dashboard

1. Click **Start Camera** to begin live detection
2. The system will detect people and check if they enter the restricted zone
3. When an alert is triggered, it will be shown in the "Recent Alerts" section
4. Click **View All Alerts** to see complete alert history
5. Click **Stop Camera** to stop the live feed

## Configuration

You can modify settings in `app/config.py`:

- `MODEL_PATH`: Path to YOLOv8 model file
- `CONFIDENCE_THRESHOLD`: Minimum confidence for detection (0.0-1.0)
- `ALERT_COOLDOWN`: Cooldown time in seconds between alerts
- `RESTRICTED_ZONE`: Coordinates of the restricted zone polygon

## Evaluation Metrics

After training, you'll find the following in the `ML/` directory:
- Trained models: `yolov8n_25epochs.pt`, `yolov8n_35epochs.pt`
- Evaluation metrics: `yolov8n_25epochs_metrics.json`, `yolov8n_35epochs_metrics.json`
- ROC curves: `yolov8n_25epochs_roc_curve.png`, `yolov8n_35epochs_roc_curve.png`

## Future Enhancements

- Add user authentication with JWT
- Integrate MongoDB for better data persistence
- Add email/SMS alert notifications
- Support multiple cameras
- Add cloud storage for evidence

## License

This project is created for educational purposes as a Final Year Project.
