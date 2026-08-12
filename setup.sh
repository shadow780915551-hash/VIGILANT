#!/usr/bin/env bash
# Streamlit Community Cloud pre-install hook.
# Runs BEFORE `pip install -r requirements.txt`.

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUMMY_DIR="${APP_DIR}/_dummy_opencv"

echo "[setup.sh] Installing dummy opencv-python package to satisfy ultralytics dep..."
echo "[setup.sh] Dummy dir: ${DUMMY_DIR}"

python -m pip install --no-input --no-deps --disable-pip-version-check -e "${DUMMY_DIR}"

echo "[setup.sh] Dummy opencv-python installed. Only opencv-python-headless from requirements.txt"
echo "[setup.sh] will provide the real cv2 runtime files — no GUI-linked build gets pulled in."
