import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# --- Configuration ---
MODEL_PATH = '/content/runs/detect/train/weights/best.pt' # Path to your trained YOLOv8 model

# Define the class names based on your training data
CLASS_NAMES = ['with helmet', 'without helmet', 'rider', 'number plate']

# Define colors for bounding boxes (matching your plotting colors if possible)
CLASS_COLORS = {
    'with helmet': (0, 255, 128),
    'without helmet': (255, 51, 51),
    'rider': (51, 255, 255),
    'number plate': (224, 102, 255)
}

# --- Streamlit App --- 
st.set_page_config(page_title="YOLOv8 Object Detection App", page_icon=":camera:")

st.title("🏍️ YOLOv8 Helmet & Rider Detection")
st.write("Upload an image to detect helmets, riders, and number plates using a custom-trained YOLOv8 model.")

@st.cache_resource # Cache the model to avoid reloading on each interaction
def load_model():
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Detecting objects...")

        # Perform inference
        results = model.predict(img_bgr, conf=0.25) # Adjust confidence threshold as needed

        # Process results and draw bounding boxes
        for r in results:
            im_bgr = r.orig_img.copy()
            if r.boxes:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    label = CLASS_NAMES[cls]
                    color = CLASS_COLORS.get(label, (0, 255, 0)) # Default to green if color not found

                    # Draw rectangle
                    cv2.rectangle(im_bgr, (x1, y1), (x2, y2), color, 2)

                    # Put label
                    text = f"{label} {conf:.2f}"
                    cv2.putText(im_bgr, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Convert back to RGB for displaying in Streamlit
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            st.image(im_rgb, caption='Detected Objects', use_column_width=True)
            st.success("Detection complete!")
    else:
        st.info("Please upload an image to start detection.")
