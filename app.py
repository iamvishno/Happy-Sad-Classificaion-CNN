import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/imageclassifier.h5")

model = load_model()


st.title("Emotion Classifier")
st.write("Upload an image to detect Happy or Sad expression")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    
    # Preprocess and predict
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)

    emotion = "SAD" if prediction[0][0] > 0.5 else "HAPPY"

    
    # Display result
    st.subheader("Result")
    st.header(f"Emotion: :red[{emotion}]")
    
 