import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

model = tf.keras.models.load_model('model.h5')

st.title("Deteksi Ekspresi Wajah")

uploaded_file = st.file_uploader("Upload gambar wajah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    img = image.resize((48, 48))  # asumsi input model 48x48
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_array = img_array.reshape(1, 48, 48, 1) / 255.0

    prediction = model.predict(img_array)
    kelas = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Kaget', 'Netral']
    hasil = kelas[np.argmax(prediction)]

    st.subheader(f"Ekspresi terdeteksi: {hasil}")
