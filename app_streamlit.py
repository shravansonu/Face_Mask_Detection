import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("model/face_mask_detector.keras")

st.title("Face Mask Detection")

uploaded = st.file_uploader("Upload Image")

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img, (224,224)) / 255.0

    box, cls = model.predict(np.expand_dims(img_resized, 0))
    h, w, _ = img.shape

    xmin, ymin, xmax, ymax = (box[0] * [w,h,w,h]).astype(int)
    cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 2)

    st.image(img, channels="BGR")
