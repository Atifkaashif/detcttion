import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Load YOLO model
model = YOLO("best.pt")

st.set_page_config(page_title="Helmet Detection", layout="wide")

st.title("🪖 Helmet Detection App")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Convert image to RGB (VERY IMPORTANT)
    image = image.convert("RGB")

    # Show uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Create temp jpg file
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    # Save safely
    image.save(temp_file.name, format="JPEG")

    # Run YOLO prediction
    results = model(temp_file.name)

    # Plot result
    result_img = results[0].plot()

    # Show result
    st.image(result_img, caption="Detection Result", use_container_width=True)
