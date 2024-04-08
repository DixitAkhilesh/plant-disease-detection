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
    url = 'http://127.0.0.1:5000/send-data'
    
    # Data to be sent
    data = {
        'disease': class_name,
        'confidence': confidence
    }
    
    try:
        response = requests.post(url, json=data)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
        
with open( "./styles.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Load the pre-trained model and class indices from files
MODEL_PATH = './model.h5'
CLASS_INDICES_PATH = './class_indices.json'
model = keras.models.load_model(MODEL_PATH, compile=False)

with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices = json.load(f)

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image for model prediction."""
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image_class(image_path):
    """Predict the class of an image using the pre-trained model."""
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    class_index = np.argmax(predictions, axis=1)[0]
    class_name = class_indices[str(class_index)]
    confidence = predictions[0][class_index] * 100  # Convert to percentage
    return class_name, round(confidence, 2)

col_title, col_uploader, col_results = st.columns([1, 1.5, 1.5])

with col_title:
    st.markdown("")
    st.markdown("## Plant Disease Detector")

with col_uploader:
    # File uploader in the second column
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
# Initialize a variable to hold the button press state outside of the columns context
button_pressed = False

# Check if an image has been uploaded before showing the "Examine" button
if uploaded_file is not None:
    with col_results:
        # Display the uploaded image immediately after uploading
        if uploaded_file is not None:
            # To display an image, you need to use st.image
            # You can adjust the width to fit your layout as needed
            st.image(uploaded_file, caption='Uploaded Image', width=250)

        button_pressed = st.button("Examine")

    # Once the button is pressed, show the results in the same column as the button
    if button_pressed:
        with st.spinner(text='Classifying...'):
            class_name, confidence = predict_image_class(uploaded_file)
        
        # Display the classification results depending on the classification
        if confidence >= 33:
            col_results.success(f"Disease: {class_name}")
            col_results.write(f"Confidence: {confidence}%")
            send_data_to_backend(class_name, confidence)
        else:
            col_results.error("Unable to detect the plant. Please provide a clearer image.")