import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import os
import time
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Helmet & Plate Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CHECK MODEL =================
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("❌ best.pt file nahi mili.")
    st.stop()

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ================= CSS =================
st.markdown("""
<style>
.main {background-color:#0e1117;}
.stMetric {background:#161b22;padding:10px;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("🪖 Safety AI")
page = st.sidebar.selectbox(
    "Choose Mode",
    ["Image Detection", "Video Detection", "Live Camera Feed"]
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 1.0, 0.4
)

# ================= FUNCTION =================
def process_frame(frame, conf):
    results = model.predict(frame, conf=conf, imgsz=640, verbose=False)
    annotated = results[0].plot()
    return annotated

# ================= TITLE =================
st.title(f"🚀 {page}")

# ==================================================
# IMAGE MODE
# ==================================================
if page == "Image Detection":

    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file).convert("RGB")
        img_np = np.array(img)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Original", use_container_width=True)

        with col2:
            output = process_frame(img_np, confidence_threshold)
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            st.image(output, caption="Detection", use_container_width=True)

# ==================================================
# VIDEO MODE
# ==================================================
elif page == "Video Detection":

    file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        if not cap.isOpened():
            st.error("Video open nahi ho rahi.")
        else:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.resize(frame, (900, 500))

                output = process_frame(frame, confidence_threshold)
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

                stframe.image(output, channels="RGB", use_container_width=True)

            cap.release()
            st.success("✅ Video Detection Complete")

# ==================================================
# LIVE CAMERA MODE
# ==================================================
elif page == "Live Camera Feed":

    st.subheader("📷 PC Camera Live Detection")

    run = st.checkbox("Start Camera")

    frame_window = st.image([])

    if run:

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error("Camera open nahi ho raha.")
        else:
            while run:
                ret, frame = cap.read()

                if not ret:
                    st.warning("Frame receive nahi hua.")
                    break

                output = process_frame(frame, confidence_threshold)
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

                frame_window.image(output, channels="RGB")

                time.sleep(0.01)

            cap.release()

    else:
        st.info("Camera OFF")
