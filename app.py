import streamlit as st
import os
import urllib.request
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# ✅ Replace this with your actual GitHub Releases download URL
MODEL_URL = 'https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0/YOUR_MODEL.h5'
MODEL_PATH = 'brain_tumor_model.h5'

# Download model if not exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Downloading model...'):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            st.success('Model downloaded successfully!')

# Preprocess uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust size based on your model input
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize
    img_array = img_array.reshape(1, 224, 224, 3)
    return img_array

# Load model and predict
def predict(image):
    img_array = preprocess_image(image)
    model = load_model(MODEL_PATH)
    prediction = model.predict(img_array)
    return prediction

# Streamlit App UI
st.title("🧠 Brain Tumor Detection from MRI")
st.write("Upload an MRI image to check for brain tumor.")

# Download model first
download_model()

# Image uploader
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🧠 Predict"):
        prediction = predict(image)
        result = "✅ No Tumor Detected" if prediction[0][0] < 0.5 else "⚠️ Tumor Detected"
        st.subheader(f"Result: {result}")
        st.write(f"Prediction Confidence: {prediction[0][0]:.4f}")
