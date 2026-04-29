import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import time

# ================= LOAD MODEL =================
model = YOLO("best.pt")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Helmet Detect",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CSS =================
st.markdown("""
<style>
.stApp{
    background:#0f1117;
    color:white;
}
section[data-testid="stSidebar"]{
    background:#111111;
    width:280px !important;
}
section[data-testid="stSidebar"] *{
    color:white !important;
}
.sidebar-title{
    text-align:center;
    font-size:28px;
    font-weight:bold;
}
.sidebar-sub{
    text-align:center;
    color:#999;
    margin-bottom:20px;
}
.metric-box{
    background:#161616;
    padding:15px;
    border-radius:10px;
    text-align:center;
    border:1px solid #222;
    margin-bottom:10px;
}
.metric-value{
    font-size:28px;
    font-weight:bold;
}
.metric-label{
    font-size:12px;
    color:#999;
}
.main-title{
    font-size:32px;
    color:#00c6ff;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:

    st.markdown('<div class="sidebar-title">HELMET DETECT</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Safety Monitor</div>', unsafe_allow_html=True)

    page = st.radio(
        "Detection Modes",
        ["📷 Detect Image", "🎥 Process Video", "📹 Webcam"]
    )

    st.markdown("---")
    st.markdown("### Live Stats")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="metric-box">
        <div class="metric-value" style="color:#00ff99;">12</div>
        <div class="metric-label">FPS</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-box">
        <div class="metric-value" style="color:#3b82f6;">183</div>
        <div class="metric-label">Frames</div>
        </div>
        """, unsafe_allow_html=True)

# ================= MAIN =================
st.markdown('<div class="main-title">🪖 Helmet & Number Plate Detection</div>', unsafe_allow_html=True)
st.write("")

# =====================================================
# IMAGE DETECTION
# =====================================================
if page == "📷 Detect Image":

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        image = Image.open(file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original", use_container_width=True)

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp.name)

        results = model.predict(
            source=temp.name,
            imgsz=640,
            conf=0.4,
            verbose=False
        )

        output = results[0].plot()

        with col2:
            st.image(output, caption="Detected", use_container_width=True)

# =====================================================
# VIDEO DETECTION (FIXED + FAST)
# =====================================================
elif page == "🎥 Process Video":

    file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if file:

        st.success("Video Uploaded Successfully")

        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(file.read())

        cap = cv2.VideoCapture(temp_video.name)

        frame_area = st.empty()
        fps_area = st.empty()

        frame_count = 0
        start_time = time.time()

        # Skip frames for speed
        skip_frames = 2

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Skip alternate frames = faster
            if frame_count % skip_frames != 0:
                continue

            # Resize for speed
            frame = cv2.resize(frame, (640, 480))

            # Detection
            results = model.predict(
                source=frame,
                imgsz=640,
                conf=0.4,
                verbose=False
            )

            frame = results[0].plot()

            frame_area.image(frame, channels="BGR", use_container_width=True)

            # FPS
            fps = frame_count / (time.time() - start_time)
            fps_area.info(f"FPS: {fps:.2f}")

        cap.release()

# =====================================================
# WEBCAM DETECTION (FAST)
# =====================================================
elif page == "📹 Webcam":

    run = st.checkbox("Start Camera")

    frame_window = st.image([])

    cap = cv2.VideoCapture(0)

    while run:

        ret, frame = cap.read()

        if not ret:
            st.error("Camera not found")
            break

        frame = cv2.resize(frame, (640,480))

        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.4,
            verbose=False
        )

        frame = results[0].plot()

        frame_window.image(frame, channels="BGR")

    cap.release()
