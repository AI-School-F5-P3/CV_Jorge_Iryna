import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
import os
from PIL import Image
import time

def setup_face_recognition(model_name):
    try:
        DeepFace.build_model(model_name)
        st.success(f"{model_name} model loaded successfully")
    except Exception as e:
        st.error(f"Error loading {model_name} model: {e}")

def verify_face(frame, reference_dir, model_name, confidence_threshold=0.6):
    try:
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return False, None, 0, False
        
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
                person_name = identity.split("/")[-2]
                return True, person_name, confidence, True
        
        return False, None, 0, True
        
    except Exception as e:
        st.error(f"Verification error: {str(e)}")
        return False, None, 0, False

def main():
    st.title("Face Recognition System")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    model_options = ["VGG-Face", "Facenet", "ArcFace", "Facenet512", "DeepFace"]
    model_name = st.sidebar.selectbox("Select Model", model_options, index=0)
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05
    )

    reference_dir = st.sidebar.text_input(
        "Reference Directory",
        value="dataset_preprocessed/iryna"
    )
    
    if not os.path.exists(reference_dir):
        st.sidebar.warning("Reference directory does not exist!")

    if st.sidebar.button("Initialize Model"):
        setup_face_recognition(model_name)

    st.header("Live Face Recognition")
    
    run = st.checkbox("Start Face Recognition")
    
    FRAME_WINDOW = st.image([])
    
    if run:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not open webcam")
            return
            
        recent_results = []
        smoothing_window = 5
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame")
                break
                
            is_match, person_name, confidence, face_detected = verify_face(
                frame, 
                reference_dir, 
                model_name=model_name,
                confidence_threshold=confidence_threshold
            )

            col1, col2 = st.columns(2)
            
            with col1:
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
                        st.success(f"Access Granted: {person_name}")
                        st.info(f"Confidence: {confidence:.2f}")
                    else:
                        cv2.putText(frame, "Access Denied", (50, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        st.error("Access Denied - Face Not Recognized")
                else:
                    recent_results.clear()
                    cv2.putText(frame, "No Face Detected", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    st.warning("No Face Detected")
            
            with col2:
                st.text(f"Model: {model_name}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            
            time.sleep(0.1)
        
        cap.release()

if __name__ == "__main__":
    main()