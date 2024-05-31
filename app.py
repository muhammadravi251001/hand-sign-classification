import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

@st.cache_data()
def load_model():
    model = tf.keras.models.load_model('rock_paper_scissors_model.h5')
    return model

def predict_class(image, model):
    image = image.resize((128, 128)) 
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    class_idx = np.argmax(predictions)

    class_names = ['paper', 'rock', 'scissors']
    return class_names[class_idx]

st.title("Image Classification: Paper, Rock, Scissors")

uploaded_file = st.file_uploader("Choose picture...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image = image.convert('RGB')
    st.image(image, use_column_width=True)
    
    st.write("")
    st.write("Classify...")

    model = load_model()

    prediction = predict_class(image, model)
    
    st.write(f"Prediction result: **{prediction}**")