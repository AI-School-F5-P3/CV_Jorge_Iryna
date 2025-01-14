#app_interface.py
import requests
import os
import cv2
import time
import requests
import streamlit as st
from PIL import Image

BACKEND_URL = "http://localhost:5000"

st.title("Face Recognition App")

with st.sidebar:
    st.header("Build Dataset")
    name_input = st.text_input("Enter name:")
    if st.button("Capture Images (Webcam)"):
        if name_input:
            # Call build dataset endpoint
            res = requests.post(f"{BACKEND_URL}/build-dataset", data={'name': name_input})
            st.write(res.json())
        else:
            st.warning("Please enter a name first.")

    st.header("Train Model")
    if st.button("Train"):
        res = requests.get(f"{BACKEND_URL}/train-model")
        st.write(res.json())

st.header("Real-Time Recognition")

method = st.selectbox("Choose recognition method", ["Upload an image", "Use webcam"])

if method == "Upload an image":
    upload_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if upload_image is not None:
        files = {'image': upload_image}
        res = requests.post(f"{BACKEND_URL}/recognize", files=files)
        result = res.json()
        st.write(f"Recognized Name: {result['name']}")
else:
    if st.button("Capture & Recognize from Webcam"):
        res = requests.get(f"{BACKEND_URL}/capture-and-recognize")
        result = res.json()
        st.write(f"Recognized Name (Webcam): {result['name']}")

st.info("Use 'Capture Images (Webcam)' in the sidebar to build the dataset and then 'Train'. After that, upload images for recognition.")