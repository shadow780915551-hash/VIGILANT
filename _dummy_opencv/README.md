# opencv-python (dummy)

Placeholder package installed BEFORE `requirements.txt` via `setup.sh`.

Only purpose: satisfy `ultralytics`'s hard dependency on the `opencv-python`
distribution name during pip/uv dependency resolution.

The real working OpenCV comes from `opencv-python-headless` in requirements.txt
— it provides the `import cv2` runtime files without Qt/GL/GUI linkage.
