# AI Posture & Presence Coach

A real-time webcam app that watches your posture, eye contact, body language, and hand gestures using your computer's camera — and gives instant on-screen warnings when something's off. Built for students, employees, and anyone practicing their presence on camera.

## What It Detects
- Posture: slouching, bending forward, bent neck, uneven shoulders, leaning sideways
- Eye contact: looking away, looking down/up, head tilt
- Body language: arms crossed, hands in pockets, fidgeting
- Face touching
- Whether you're visible in frame
- Hand gestures (optional, off by default) — works for multiple people at once

## How It Works
It calibrates to YOUR normal posture and gaze for the first ~1.5 seconds (sit straight, look at camera), then watches for changes from that baseline. When something's wrong, a big warning banner shows up right on the video — impossible to miss.

## Setup
1. Clone this repo
2. Create a virtual environment:
   python -m venv venv
   venv\Scripts\activate     (Windows)
   source venv/bin/activate  (Mac/Linux)
3. Install requirements:
   pip install -r requirements.txt
4. Run it:
   python main.py
5. Sit straight and look at the camera to calibrate. Press 'c' to recalibrate, 'q' to quit.

## Extra Features
- Saves a report after every session (posture %, eye contact %, confidence score)
- Logs every alert to a database (incidents.db)
- Run `python analytics.py` to see your progress over time
- Optional email alerts for serious incidents (off by default)
- Works better in low light with built-in image enhancement

## Tech Used
Python, OpenCV, MediaPipe (Pose + FaceMesh + Hands), SQLite

## Limitations
- Eye contact tracking is a smart approximation, not true 3D gaze tracking
- Works best with one main person for posture/eye contact (hand gestures support multiple people)
- Gesture recognition is rule-based by default; can be swapped for a trained ML model

## Future Ideas
- Train a custom deep learning model for more gestures
- Full multi-person body tracking
- Web dashboard
- SMS alerts
