import json
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model= keras.models.load_model('./model.h5', compile= False)

with open('./class_indices.json', 'r') as f:
    class_indices= json.load(f)

def load_and_preprocess_image(image_path, target_size=(224,224)):
    img= Image.open(image_path)
    img= img.resize(target_size)
    img_array= np.array(img)
    img_array= np.expand_dims(img_array, axis=0)
    img_array= img_array.astype('float32')/ 255.
    return img_array

def predict_image_class(image_path):  
    preprocessed_img= load_and_preprocess_image(image_path)
    predictions= model.predict(preprocessed_img)
    predict_class_index= np.argmax(predictions, axis=1)[0]
    
    predict_class_name= class_indices[str(predict_class_index)]
    
    confidence = predictions[0][predict_class_index]  # Extracting confidence from predictions array
    
    c = round(confidence * 100, 2)
    string_confidence = str(c)
    return predict_class_name, string_confidence
