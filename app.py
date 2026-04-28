import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Load model
model = YOLO("best.pt")

st.title("Helmet Detection App")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Get file extension
    file_ext = os.path.splitext(uploaded_file.name)[1]

    # Create temp file with extension
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)

    # Save image correctly
    image.save(temp_file.name)

    # Predict
    results = model(temp_file.name)

    # Plot result
    result_img = results[0].plot()

    st.image(result_img, caption="Detection Result", use_container_width=True)
