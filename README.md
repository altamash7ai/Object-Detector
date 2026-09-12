# Real-Time Object Detection with YOLOv8

A Python computer vision project that detects objects in images, video files, and live webcam feed using YOLOv8 and OpenCV. Tracks detection counts per object type and generates a summary report after each session.


## Features

- Detects 80 common object classes (person, car, phone, chair, etc. — full COCO dataset)
- Works on three input types: static images, video files, and live webcam
- Real-time FPS counter to monitor performance
- Automatic object counting across a session
- Generates a `.txt` detection report after each run (object types + counts, timestamped)

## Why YOLOv8

This project originally used SSD MobileNet v3, a lighter but less accurate model. During testing, it frequently confused visually similar handheld objects (e.g. mistaking a phone for a knife or scissors). After investigating the tradeoffs between detection speed and accuracy, I switched to YOLOv8, which uses a stronger backbone and significantly reduced this misclassification, at the cost of somewhat higher compute per frame.

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLOv8
- COCO pretrained weights (80 object classes)

## Installation

```bash
git clone https://github.com/altamash7ai/Object-Detector.git
cd Object-Detector
pip install -r requirements.txt
```

## Usage

The script (`object_detector.py`) runs image detection first, then video/webcam detection. Two lines control what it detects — edit these before running:

**1. Image detection — change the image path:**
```python
img = cv2.imread('boyyy.jpg')   # replace with your own image filename/path
```

**2. Video/webcam detection — change the source in `cv2.VideoCapture(...)`:**

If you want to use your **live camera**, pass `0`:
```python
cap = cv2.VideoCapture(0)
```

If you want to detect on a **video file** instead, pass the file path as a string:
```python
cap = cv2.VideoCapture('video.mp4')   # replace with your own video filename/path
```

Then run:
```bash
python object_detector.py
```

Press `q` at any time to stop video/webcam detection early. A detection report (`detection_report_<timestamp>.txt`) is automatically saved in the project folder once the session ends.

## Sample Report Output

```
Object Detection Report
Generated: 2026-09-12 14:30:00
------------------------------
Total unique object types detected: 3

Detections (frame-count basis):
  person: 142
  cell phone: 58
  chair: 12
```

## What I Learned

- How single-stage object detectors (YOLO) differ architecturally from region-based detectors (SSD)
- The real-world tradeoff between model size, speed (FPS), and accuracy
- Debugging a live OpenCV + ML pipeline (camera backend issues, confidence threshold tuning, preprocessing bugs)
- Why lightweight pretrained models struggle with fine-grained visual distinctions, and when fine-tuning becomes necessary
- Using Git and GitHub to version and publish a project

## Possible Improvements

- Object tracking (count unique objects across frames instead of per-frame appearances) using `model.track()`
- Fine-tuning on a custom dataset for objects the pretrained model confuses
- CSV export of detection reports for further analysis
- Command-line arguments for input source selection (webcam/video/image) instead of editing code directly

## Author

Built by Altamash as a hands-on project to learn computer vision fundamentals with OpenCV and YOLOv8.
