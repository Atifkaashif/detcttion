import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import os
import time

# ================= PAGE =================
st.set_page_config(
    page_title="Helmet Detection",
    layout="wide"
)

# ================= CHECK MODEL =================
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("best.pt model file missing")
    st.stop()

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ================= CSS =================
st.markdown("""
<style>
.stApp{
background:#0f1117;
color:white;
}
section[data-testid="stSidebar"]{
background:#111111;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("🪖 Helmet Detect")
page = st.sidebar.radio(
    "Select Mode",
    ["Image Detection", "Video Detection", "Live Camera"]
)

# ================= TITLE =================
st.title("🪖 Helmet Detection System")

# ===================================================
# IMAGE
# ===================================================
if page == "Image Detection":

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:

        img = Image.open(file).convert("RGB")

        col1,col2 = st.columns(2)

        with col1:
            st.image(img, caption="Original")

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        img.save(temp.name)

        results = model.predict(
            source=temp.name,
            conf=0.4,
            imgsz=640,
            verbose=False
        )

        output = results[0].plot()

        with col2:
            st.image(output, caption="Detected")

# ===================================================
# VIDEO
# ===================================================
elif page == "Video Detection":

    file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if file:

        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(file.read())

        cap = cv2.VideoCapture(temp.name)

        frame_box = st.empty()
        fps_box = st.empty()

        count = 0
        start = time.time()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            count += 1

            # Skip frames for speed
            if count % 2 != 0:
                continue

            frame = cv2.resize(frame, (640,480))

            results = model.predict(
                source=frame,
                conf=0.4,
                imgsz=640,
                verbose=False
            )

            frame = results[0].plot()

            frame_box.image(frame, channels="BGR")

            fps = count / (time.time() - start)
            fps_box.info(f"FPS: {fps:.2f}")

        cap.release()

# ===================================================
# CAMERA
# ===================================================
elif page == "Live Camera":

    run = st.checkbox("Start Camera")

    FRAME = st.image([])

    cap = cv2.VideoCapture(0)

    while run:

        ret, frame = cap.read()

        if not ret:
            st.error("Camera not found")
            break

        frame = cv2.resize(frame, (640,480))

        results = model.predict(
            source=frame,
            conf=0.4,
            imgsz=640,
            verbose=False
        )

        frame = results[0].plot()

        FRAME.image(frame, channels="BGR")

    cap.release()
