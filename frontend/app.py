#app_interface.py
import requests
import os
import cv2
import time
import requests
import streamlit as st
from PIL import Image
from deepface import DeepFace
import numpy as np


# Define root path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Update paths using ROOT_DIR
DATASET_PATH = os.path.join(ROOT_DIR, 'backend', 'dataset')
DATASET_PREPROCESSED_PATH = os.path.join(ROOT_DIR, 'backend', 'dataset_preprocessed')
MODELS_PATH = os.path.join(ROOT_DIR, 'backend', 'models')


def get_model_selection():

    models = {
        "VGG-Face": "Traditional model, good baseline",
        "Facenet": "Good balance of speed and accuracy",
        "ArcFace": "Very accurate but slower",
        "Facenet512": "More accurate version of Facenet",
        "DeepFace": "Alternative implementation"
    }
    
    # Create selection box with model names
    selected_model = st.selectbox(
        "Select Face Recognition Model",
        list(models.keys()),
        index=list(models.keys()).index("Facenet512"),
        help="Hover over options to see model descriptions"
    )
    
    # Show description for selected model
    st.caption(models[selected_model])
    
    return selected_model

def capture_image(name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return False

    cv2.imshow('Capture', frame)
    img_name = f"{name}_{time.time()}.jpg"
    try:
        cv2.imwrite(os.path.join(DATASET_PATH, name, img_name), frame)
    except Exception as e:
        print(f"Error saving image: {e}")
        cap.release()
        cv2.destroyAllWindows()
        return False

    cap.release()
    cv2.destroyAllWindows()
    return True


def setup_face_recognition(model_name):

    try:
        DeepFace.build_model(model_name)
        st.write(f"{model_name} model loaded successfully")
    except Exception as e:
        st.write(f"Error loading {model_name} model: {e}")

def verify_face(frame, reference_dir, model_name, confidence_threshold=0.6):

    try:
        # First check if any face is detected
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangle around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(
            frame, 
            (x, y),                   # Start point
            (x + w, y + h),          # End point
            (255, 0, 0),             # Color (Blue)
            2                       # Thickness
            )    


        if len(faces) == 0:
            return False, None, 0, False
        
        # Perform face recognition
        result = DeepFace.find(
            img_path=frame,
            db_path=reference_dir,
            enforce_detection=False,
            model_name=model_name,
            distance_metric="cosine"
        )
        
        if len(result[0]) > 0:
            matched_face = result[0].iloc[0]
            distance_column = f"{model_name}_cosine"
            distance = matched_face.get(distance_column, matched_face.get('distance', 1.0))
            
            confidence = 1 - float(distance)
            
            if confidence > confidence_threshold:
                identity = matched_face['identity']
                person_name = identity.split("/")[-1]
                person_name = person_name.split("\\")[0]
                return True, person_name, confidence, True
        
        return False, None, 0, True
        
    except Exception as e:
        print(f"Verification error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, 0, False


def run_webcam_recognition(reference_dir, model_name):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Error: Could not open webcam")
        return

    # Create placeholder for video feed
    video_placeholder = st.empty()
    
    # Create stop button
    stop_button = st.button("Stop Recognition")
    
    recent_results = []
    smoothing_window = 5
    
    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame")
            break
            
        is_match, person_name, confidence, face_detected = verify_face(
            frame, 
            reference_dir, 
            model_name=model_name
        )
        
        if face_detected:
            recent_results.append(is_match)
            if len(recent_results) > smoothing_window:
                recent_results.pop(0)
                
            access_granted = sum(recent_results) > len(recent_results) / 2
            
            if access_granted and person_name:
                cv2.putText(frame, f"Access Granted: {person_name}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (50, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Access Denied - Face Not Recognized", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            recent_results.clear()
            cv2.putText(frame, "No Face Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(frame, f"Model: {model_name}", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Convert BGR to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Update the video feed
        video_placeholder.image(frame_rgb)

    cap.release()
    st.success("Recognition stopped")


### MAIN APP ###


reference_dir = "dataset_preprocessed/jorge"

st.title("Face Recognition App")

with st.sidebar:
    #Sección para capturar imágenes
    st.header("Build Dataset")
    name_input = st.text_input("Enter name:")
    if st.button("Capture Image (Webcam)"):
        if name_input:
            succeeded = capture_image(name=name_input)
            if succeeded:
                st.write("Image captured.")
            else:
                st.write("Failed to capture images.")
        else:
            st.warning("Please enter a name first.")


    # Replace existing model_name assignment with:
    model_name = get_model_selection()

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
        setup_face_recognition(model_name)
        run_webcam_recognition(os.path.join(DATASET_PATH, name_input), model_name)

st.info("Use 'Capture Images (Webcam)' in the sidebar to build the dataset. After that, use webcam or upload image for recognition.")