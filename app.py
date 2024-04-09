import streamlit as st
import json
import numpy as np
from PIL import Image
from tensorflow import keras
import os
import requests  # Import the requests library

st.set_page_config(layout="wide")

def send_data_to_backend(class_name, confidence):
    """Send classification results to the backend."""
    url = 'https://streamlit-node-connector.onrender.com/send-data'
    data = {'disease': class_name, 'confidence': confidence}
    try:
        response = requests.post(url, json=data)
        if response.status_code != 200:
            st.error("Failed to send data to the backend.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Load CSS styles
with open("./styles.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

# Suppress TensorFlow logging and disable oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the pre-trained model and class indices from files
MODEL_PATH = './model.h5'
CLASS_INDICES_PATH = './class_indices.json'
model = keras.models.load_model(MODEL_PATH, compile=False)

with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices = json.load(f)

def load_and_preprocess_image(image_file, target_size=(224, 224)):
    """Load and preprocess an image for model prediction."""
    img = Image.open(image_file)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image_class(image_file):
    """Predict the class of an image using the pre-trained model."""
    preprocessed_img = load_and_preprocess_image(image_file)
    predictions = model.predict(preprocessed_img)
    class_index = np.argmax(predictions, axis=1)[0]
    class_name = class_indices[str(class_index)]
    confidence = predictions[0][class_index] * 100  # Convert to percentage
    return class_name, round(confidence, 2)

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: whitesmoke !important;
    margin-top: 0;
    color: black;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: whitesmoke;
    border: 2px solid #0c330b;
    color: black;
}
[data-testid="baseButton-secondary"] {
    background-color: #0c330b;
    color: white;
    border: 1px solid #0c330b;
}
.e1bju1570 {
    color: gray;
}
[data-testid="baseButton-secondary"]:focus {
    color: white;
    border: 1px solid #0c330b;
}
[data-testid="stNotification"] {
    color: black
}
</style>
"""

col_title, col_uploader, col_results = st.columns([1, 1.5, 1.5])

st.markdown(page_bg, unsafe_allow_html=True)

with col_title:
    st.markdown("## Plant Disease Detector")

with col_uploader:
    uploaded_file = st.file_uploader("", type=['jpg', 'png'], key="file_uploader")

# Handle image upload and processing
if uploaded_file is not None:
    with col_results:
        st.image(uploaded_file, caption='Uploaded Image', width=250)
        
        if st.button("Examine"):
            class_name, confidence = predict_image_class(uploaded_file)
            
            if confidence >= 33:
                col_results.success(f"Disease: {class_name}, Confidence: {confidence}%")
                send_data_to_backend(class_name, confidence)
            else:
                col_results.error("Unable to detect the plant. Please provide a clearer image.")
