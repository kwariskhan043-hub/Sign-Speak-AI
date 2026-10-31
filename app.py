
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import zipfile
import os
from PIL import Image

st.set_page_config(
    page_title="SignSpeak AI",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 SignSpeak AI")
st.subheader("Hand Sign Language Recognition Using Deep Learning")

@st.cache_resource
def load_model():
    zip_path = "artifacts/sign_language_cnn.zip"
    extract_path = "artifacts"

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    model = tf.keras.models.load_model(
        os.path.join(extract_path, "sign_language_cnn.keras")
    )

    with open("artifacts/class_names.json", "r") as f:
        class_names = json.load(f)

    return model, class_names

try:
    model, class_names = load_model()
    st.success("AI Model loaded successfully!")

    uploaded_file = st.file_uploader(
        "Upload an ASL hand sign image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Hand Sign",
                 use_container_width=True)

        img = image.resize((64, 64))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions[0]))
        predicted_sign = class_names[predicted_index]
        confidence = float(np.max(predictions[0])) * 100

        st.success(f"Predicted Sign: {predicted_sign}")
        st.write(f"Confidence: {confidence:.2f}%")

except Exception as e:
    st.error(f"Error loading model: {e}")

