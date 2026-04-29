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
    st.error(f"❌ Error: '{MODEL_PATH}' file nahi mili. Please check the file name.")
    st.stop()

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ================= CUSTOM CSS =================
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("🪖 Safety AI")
st.sidebar.markdown("---")
page = st.sidebar.selectbox(
    "Choose Input Mode",
    ["Image Detection", "Video Detection", "Live Camera Feed"]
)

st.sidebar.info("Ye system Helmet aur Number Plate detect karne ke liye banaya gaya hai.")

# ================= FUNCTIONS =================
def process_frame(frame, conf):
    # YOLO prediction
    results = model.predict(source=frame, conf=conf, imgsz=640, verbose=False)
    # Draw boxes on frame
    annotated_frame = results[0].plot()
    return annotated_frame

# ================= MAIN CONTENT =================
st.title(f"🚀 {page}")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)

# ----------------- IMAGE MODE -----------------
if page == "Image Detection":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if file:
        img = Image.open(file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Original")
            st.image(img, use_container_width=True)
            
        with col2:
            st.header("Detection Results")
            # Convert PIL to OpenCv for processing if needed, or pass directly
            res_img = process_frame(img, confidence_threshold)
            st.image(res_img, use_container_width=True)

# ----------------- VIDEO MODE -----------------
elif page == "Video Detection":
    file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty() # Placeholder for video frames
        
        stop_btn = st.button("Stop Video")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_btn:
                break
                
            # Resize for speed
            frame = cv2.resize(frame, (850, 480))
            
            # Process and Plot
            output_frame = process_frame(frame, confidence_threshold)
            
            # Convert BGR (OpenCV) to RGB (Streamlit)
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(output_frame, channels="RGB", use_container_width=True)
            
        cap.release()
        st.success("Video Processing Complete")

# ----------------- LIVE CAMERA -----------------
# ================= LIVE CAMERA FEED =================
elif page == "Live Camera Feed":
    st.subheader("Webcam Live Detection")
    run_cam = st.checkbox("Turn On Camera")
    
    FRAME_WINDOW = st.image([])
    
    # Session state to manage camera object
    if run_cam:
        # 0 index default camera ke liye hota hai
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            st.error("Error: System camera ko access nahi kar pa raha. Check permissions.")
        
        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.warning("Frame read nahi ho raha. Camera busy ho sakta hai.")
                break
            
            # Process frame using your YOLO model
            output_frame = process_frame(frame, confidence_threshold)
            
            # Convert for Streamlit display
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(output_frame, channels="RGB")
            
            # Small delay to keep UI responsive
            time.sleep(0.01)
            
        cap.release()
    else:
        st.info("Camera is currently OFF.")
