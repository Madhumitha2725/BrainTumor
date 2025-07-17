import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Title
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Detection")
st.markdown("Upload an MRI image and the model will predict whether it's **Tumor** or **Normal**.")

# File ID from Google Drive (replace with your actual ID)
file_id = "1xTkOky1ji1iXkjHt8pKldreNZMA9LQim"  # e.g., "1XyzABC12345"
model_path = "brain_tumor_model.h5"

# Download the model if not already present
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        url = f"https://drive.google.com/uc?id=1xTkOky1ji1iXkjHt8pKldreNZMA9LQim"
        gdown.download(url, model_path, quiet=False)

# Load the model
model = load_model(model_path)

# File uploader
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

# Prediction
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded MRI", use_column_width=True)

    # Preprocess
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_label = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"

    # Show result
    st.markdown("---")
    st.subheader("🩺 Prediction Result")
    st.success(predicted_label)
