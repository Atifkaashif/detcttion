import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

# Load model
from pathlib import Path
model_path = Path(__file__).parent / "best.pt"
model = YOLO(str(model_path))

st.set_page_config(page_title="Helmet Detection Live", layout="centered")

st.title("🪖 Live Helmet Detection System")

run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image()

# Video writer setup
recording = False
out = None

if st.button("🎥 Start Recording"):
    recording = True
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    st.success("Recording Started...")

if st.button("⏹ Stop Recording"):
    recording = False
    if out:
        out.release()
    st.warning("Recording Stopped!")

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Camera not found")
        break

    # YOLO prediction
    results = model(frame)
    annotated_frame = results[0].plot()

    # Convert BGR → RGB
    image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Show live frame
    FRAME_WINDOW.image(image)

    # Save recording if ON
    if recording and out:
        out.write(annotated_frame)

    time.sleep(0.03)

camera.release()
if out:
    out.release()
