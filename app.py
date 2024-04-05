import streamlit as st
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
import os

# Suppress TensorFlow and Keras warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the pre-trained model
model = keras.models.load_model('./model.h5', compile=False)

# Load class indices
with open('./class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to predict image class
def predict_image_class(image_path):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predict_class_index = np.argmax(predictions, axis=1)[0]
    predict_class_name = class_indices[str(predict_class_index)]

    confidence = predictions[0][predict_class_index]  # Extracting confidence from predictions array
    percentage = round(confidence * 100, 2)
    return predict_class_name, percentage

# Streamlit app title and styling
st.title("Plant Disease Detector")
st.markdown("---")
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 8px 14px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            border-radius: 4px;
            border: none;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>div>div>input {
            border-radius: 4px;
            border: 1px solid #ccc;
            padding: 6px 10px;
            width: 100%;
            box-sizing: border-box;
            margin-top: 4px;
            margin-bottom: 8px;
        }
        .stTextInput>div>div>input:focus {
            outline: none;
            border-color: #4CAF50;
        }
        .stImage>img {
            border-radius: 8px;
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.1), 0 6px 20px 0 rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# File uploader and classification button
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', width=300)
    st.write("")
    
    classify_button = st.button("Examine")

    if classify_button:
        with st.spinner(text='Classifying...'):
            class_name, confidence = predict_image_class(uploaded_file)
        
        if confidence >= 33:
            st.success(f"Disease: {class_name}")
            st.write(f"Confidence: {confidence}%")
        else:
            st.error("Unable to detect the plant. Please provide a clearer image.")
