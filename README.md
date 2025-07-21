# Real-Time Object Tracker

This project implements a real-time object tracker using OpenCV. It allows users to select an object in the webcam feed and track it live using CPU or GPU acceleration.

---

## Features

- Select any object manually in the first frame
- Real-time tracking using:
  - OpenCV Legacy Trackers (CSRT, KCF, MOSSE, MIL)
  - Template Matching (CPU or CUDA-accelerated GPU)
- Automatically detects and uses GPU if available
- Displays:
  - Bounding box and tracker name
  - FPS, CPU usage, GPU usage
  - Live status: TRACKING or TARGET LOST

---

## Requirements

Install dependencies:

```bash
pip install opencv-python numpy psutil
