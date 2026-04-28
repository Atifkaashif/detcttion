import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2

# Load model
model = YOLO("best.pt")

# Page config
st.set_page_config(
    page_title="Helmet & Number Plate Detection",
    layout="wide"
)

# ================= HEADER =================
st.markdown("""
    <style>
    .header {
        background-color: #1f4e79;
        padding: 15px;
        text-align: center;
        color: white;
        font-size: 28px;
        font-weight: bold;
        border-radius: 10px;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #1f4e79;
        color: white;
        text-align: center;
        padding: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">🪖 Helmet & Number Plate Detection System</div>', unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("📌 Menu")

page = st.sidebar.radio("Select Option", [
    "📷 Image Detection",
    "🎥 Video Detection",
    "📹 Live Camera"
])

# ================= IMAGE DETECTION =================
if page == "📷 Image Detection":

    st.subheader("Upload Image for Detection")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((400, 400))

        col1, col2 = st.columns(2)

        with col1:
            st.write("Input Image")
            st.image(image)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_file.name)

        results = model(temp_file.name)
        result_img = results[0].plot()

        result_img = Image.fromarray(result_img)
        result_img = result_img.resize((400, 400))

        with col2:
            st.write("Detection Result")
            st.image(result_img)

# ================= VIDEO DETECTION =================
elif page == "🎥 Video Detection":

    st.subheader("Upload Video")

    video_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if video_file is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            frame = results[0].plot()

            stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()

# ================= LIVE CAMERA =================
elif page == "📹 Live Camera":

    st.subheader("Live Webcam Detection")

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image()

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Camera not found")
            break

        results = model(frame)
        frame = results[0].plot()

        FRAME_WINDOW.image(frame, channels="BGR")

    camera.release()

# ================= FOOTER =================
st.markdown("""
<div class="footer">
    🚀 Developed for FYP | Helmet & Number Plate Detection System
</div>
""", unsafe_allow_html=True)
