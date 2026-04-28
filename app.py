import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("Helmet Detection System")

model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    image.save(temp_file.name)

    results = model(temp_file.name)

    result_img = results[0].plot()

    st.image(result_img, caption="Detection Result")
