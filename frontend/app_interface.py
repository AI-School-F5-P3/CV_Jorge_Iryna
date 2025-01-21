# python
import os
import cv2
import time
import logging
import streamlit as st
import requests
import pickle
from PIL import Image
from deepface import DeepFace
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(ROOT_DIR, 'backend', 'dataset')
MODELS_PATH = os.path.join(ROOT_DIR, 'backend', 'models')
MODEL_FILE = os.path.join(MODELS_PATH, 'face_model_facenet512.pkl')
STYLE_PATH = os.path.join(ROOT_DIR, 'backend', 'styles', 'style.css')
LOGO_PATH = os.path.join(ROOT_DIR, 'backend', 'styles', 'logo.png')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

st.set_page_config(
    page_title="FaceControl",
    page_icon="👽",
    layout="wide"
)
def load_logo(logo_path):
    try:
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        with col3:
            st.image(logo_path, width=150)
    except Exception as e:
        st.error(f"Error loading logo file: {e}")

load_logo(LOGO_PATH)


def load_css():
    try:
        with open(STYLE_PATH) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading CSS file: {e}")
        
load_css()

def train_or_update_model(model_name):
    """
    Build or update the face recognition model by scanning all subdirs in backend/dataset.
    Stores the resulting model in backend/models/face_model_facenet512.pkl.
    """
    logging.info("Checking if model update is needed...")
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    need_retrain = True
    if os.path.exists(MODEL_FILE):
        logging.info("Model file already exists. You can implement more checks (e.g., modified time).")
        need_retrain = False

    if need_retrain:
        try:
            logging.info(f"Building {model_name} model...")
            _ = DeepFace.build_model(model_name)  # load model into memory
            # Optionally scan dataset to force embedding creation
            # DeepFace.find(img_path='dummy.jpg', db_path=DATASET_PATH, model_name=model_name, enforce_detection=False)
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump(model_name, f)  # just storing model name for demonstration
            logging.info(f"Model saved at {MODEL_FILE}")
        except Exception as e:
            logging.error(f"Error training/updating model: {e}")


def capture_image(name):
    """Capture a single image from webcam and store it in backend/dataset/<name>."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
        return False

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return False

    folder_path = os.path.join(DATASET_PATH, name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    img_name = f"{name}_{int(time.time())}.jpg"
    save_path = os.path.join(folder_path, img_name)
    cv2.imwrite(save_path, frame)
    cap.release()
    cv2.destroyAllWindows()
    logging.info(f"Captured and saved image to {save_path}")
    return True


def setup_face_recognition(model_name):
    """
    Loads the face recognition model into memory (after ensuring it's trained).
    """
    train_or_update_model(model_name)  # ensure model is ready
    try:
        DeepFace.build_model(model_name)
        st.write(f"{model_name} model loaded.")
        logging.info(f"{model_name} model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading {model_name} model: {e}")
        logging.error(f"Error loading model: {e}")


def verify_uploaded_image(uploaded_image, model_name):
    """
    Recognize who is in the uploaded image, draw bounding boxes, and display results.
    """
    try:
        # Convert uploaded file to OpenCV image
        file_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Load face detector
        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Perform recognition
        result = DeepFace.find(
            img_path=frame,
            db_path=DATASET_PATH,
            model_name=model_name,
            distance_metric="cosine",
            enforce_detection=False
        )

        # Draw boxes and labels
        if len(result) > 0 and not result[0].empty:
            matched_face = result[0].iloc[0]
            identity_path = matched_face['identity']
            person_name = os.path.basename(os.path.dirname(identity_path))
            
            # Draw box for each detected face
            for (x, y, w, h) in faces:
                # Draw rectangle
                cv2.rectangle(
                    frame, 
                    (x, y), 
                    (x+w, y+h), 
                    (0, 255, 0), 
                    2
                )
                
                # Add name label
                cv2.putText(
                    frame,
                    f"{person_name}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )
            
            # Convert to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Recognized Face(s)")
            st.success(f"Recognized: {person_name}")
            logging.info(f"Recognized: {person_name}")
        else:
            # Still show image with boxes even if not recognized
            for (x, y, w, h) in faces:
                cv2.rectangle(
                    frame, 
                    (x, y), 
                    (x+w, y+h), 
                    (0, 0, 255), 
                    2
                )
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Unrecognized Face(s)")
            st.warning("No match found.")
            logging.info("No match found in the dataset.")
            
    except Exception as e:
        st.error(f"Error in face recognition: {str(e)}")
        logging.error(f"Face recognition error: {str(e)}")


def verify_face(frame, model_name, confidence_threshold=0.6):
    """
    Perform real-time webcam recognition. Returns (is_match, person_name, confidence, face_detected).
    """
    try:
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
        
        # Draw rectangle around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(
            frame, 
            (x, y),                   # Start point
            (x + w, y + h),          # End point
            (0, 255, 0),             # Color (Blue)
            2                       # Thickness
            )    

        if len(faces) == 0:
            return False, None, 0, False
        
        # Only process the first face found for simplicity
        result = DeepFace.find(
            img_path=frame,
            db_path=DATASET_PATH,
            model_name=model_name,
            distance_metric="cosine",
            enforce_detection=False
        )
        
        if len(result) > 0 and not result[0].empty:
            matched_face = result[0].iloc[0]
            distance_column = f"{model_name}_cosine" if f"{model_name}_cosine" in matched_face else 'distance'
            distance = matched_face.get(distance_column, 1.0)
            confidence = 1 - float(distance)
            
            if confidence > confidence_threshold:
                identity = matched_face['identity']
                person_name = os.path.basename(os.path.dirname(identity))
                return True, person_name, confidence, True
        
        return False, None, 0, True
    except Exception as e:
        logging.error(f"Verification error: {str(e)}")
        return False, None, 0, False


def run_webcam_recognition(model_name):
    """
    Continuously read from webcam, try to recognize faces, and show results in Streamlit.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
        return

    video_placeholder = st.empty()
    stop_button = st.button("Stop Recognition")

    recent_results = []
    smoothing_window = 5

    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Could not read frame from webcam.")
            break
        
        is_match, person_name, confidence, face_detected = verify_face(frame, model_name)
        
        if face_detected:
            recent_results.append(is_match)
            if len(recent_results) > smoothing_window:
                recent_results.pop(0)
            
            access_granted = sum(recent_results) > len(recent_results) / 2
            if access_granted and person_name:
                cv2.putText(frame, f"Access Granted: {person_name}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (50, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Access Denied - Face Not Recognized", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            recent_results.clear()
            cv2.putText(frame, "No Face Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb)

    cap.release()
    st.info("Recognition stopped.")
    logging.info("Webcam recognition stopped.")


############# STREAMLIT UI #############
st.title("FaceControl")

# 1) Capture images
st.sidebar.header("Build Dataset")
name_input = st.sidebar.text_input("Enter name for capturing images:")
if st.sidebar.button("Capture Image (Webcam)"):
    if name_input:
        success = capture_image(name_input)
        if success:
            st.sidebar.success(f"Image for {name_input} captured.")
        else:
            st.sidebar.error("Failed to capture.")
    else:
        st.sidebar.warning("Please enter a name first.")

# 2) Select model & set up
models = ["VGG-Face", "Facenet", "ArcFace", "Facenet512", "DeepFace"]
model_name = st.sidebar.selectbox("Select Face Recognition Model:", models, index=3)  # index=3 Defaults to Facenet512

# 3) Choose recognition method
st.header("Identity Recognition")
method = st.selectbox("Choose recognition method", ["Upload an image", "Use webcam"])

if method == "Upload an image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        setup_face_recognition(model_name)  # Ensure model is up to date
        verify_uploaded_image(uploaded_image, model_name)
else:
    if st.button("Start Webcam Recognition"):
        setup_face_recognition(model_name)
        run_webcam_recognition(model_name)

st.info("Use the sidebar to build the dataset and select the model. Then choose a recognition method above.")