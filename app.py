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

# ================= CUSTOM CSS =================
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #0f1117;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111111;
    width: 280px !important;
    border-right: 1px solid #222;
}

/* Sidebar Text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Sidebar Title */
.sidebar-title {
    text-align:center;
    font-size:28px;
    font-weight:bold;
    margin-top:10px;
}

.sidebar-sub {
    text-align:center;
    color:#999;
    font-size:13px;
    margin-bottom:20px;
}

/* Buttons */
.stButton>button {
    width:100%;
    border:none;
    border-radius:8px;
    padding:12px;
    font-weight:bold;
    color:white;
}

/* Cards */
.metric-box {
    background:#161616;
    padding:18px;
    border-radius:10px;
    text-align:center;
    margin-bottom:10px;
    border:1px solid #222;
}

.metric-value {
    font-size:30px;
    font-weight:bold;
}

.metric-label {
    font-size:13px;
    color:#999;
}

.main-title{
    font-size:32px;
    font-weight:bold;
    color:#00c6ff;
}

</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:

    st.markdown('<div class="sidebar-title">HELMET DETECT</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Safety Monitor</div>', unsafe_allow_html=True)

    st.markdown("### Detection Modes")

    page = st.radio(
        "",
        ["📷 Detect Image", "🎥 Process Video", "📹 Webcam"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Live Statistics")

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

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value" style="color:red;">2</div>
            <div class="metric-label">Violations</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value" style="color:#00ff99;">98</div>
            <div class="metric-label">Safe</div>
        </div>
        """, unsafe_allow_html=True)

# ================= MAIN AREA =================
st.markdown('<div class="main-title">🪖 Helmet & Number Plate Detection System</div>', unsafe_allow_html=True)
st.write("")

# ================= IMAGE =================
if page == "📷 Detect Image":

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp.name)

        results = model(temp.name)
        result_img = results[0].plot()

        with col2:
            st.image(result_img, caption="Detection Result", use_container_width=True)

# ================= VIDEO =================
elif page == "🎥 Process Video":

    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if video_file:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        frame_box = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            frame = results[0].plot()

            frame_box.image(frame, channels="BGR", use_container_width=True)

        cap.release()

# ================= WEBCAM =================
elif page == "📹 Webcam":

    run = st.checkbox("Start Camera")

    frame_window = st.image([])

    cam = cv2.VideoCapture(0)

    while run:
        ret, frame = cam.read()

        if not ret:
            st.error("Camera not found")
            break

        results = model(frame)
        frame = results[0].plot()

        frame_window.image(frame, channels="BGR")

    cam.release()
