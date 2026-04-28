import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Load model
model = YOLO("best.pt")

# Page config
st.set_page_config(
    page_title="Helmet Detection",
    layout="centered"
)

# Custom CSS (Web App Style)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fb;
    }
    h1 {
        text-align: center;
        color: #1f4e79;
    }
    .block-container {
        padding-top: 2rem;
    }
    img {
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🪖 Helmet Detection System")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Resize image (IMPORTANT for UI control)
    image = image.resize((400, 400))

    # Layout columns (like web app)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=False)

    # Save temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_file.name, format="JPEG")

    # Prediction
    results = model(temp_file.name)
    result_img = results[0].plot()

    # Resize output image
    result_img = Image.fromarray(result_img)
    result_img = result_img.resize((400, 400))

    with col2:
        st.subheader("Detection Result")
        st.image(result_img, use_container_width=False)
