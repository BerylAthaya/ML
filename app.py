import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model.h5")
expression_map = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("Deteksi Ekspresi Wajah 😊😡😭")

uploaded_file = st.file_uploader("Upload gambar wajah", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L").resize((48, 48))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))

    prediction = model.predict(img_array)
    label = expression_map[np.argmax(prediction)]

    st.image(image, caption="Wajah yang diunggah", width=200)
    st.write(f"Ekspresi terdeteksi: **{label}**")
