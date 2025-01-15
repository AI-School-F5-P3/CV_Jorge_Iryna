import cv2
from deepface import DeepFace
import numpy as np

def setup_face_recognition(model_name):

    try:
        DeepFace.build_model(model_name)
        print(f"{model_name} model loaded successfully")
    except Exception as e:
        print(f"Error loading {model_name} model: {e}")

def verify_face(frame, reference_dir, model_name, confidence_threshold=0.6):

    try:
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
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
        print("Error: Could not open webcam")
        return
        
    recent_results = []
    smoothing_window = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
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
        
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    reference_dir = "dataset_preprocessed/jorge"

    # model_name = "VGG-Face"    
    # model_name = "Facenet"      # Good balance of speed and accuracy
    # model_name = "ArcFace"     # Very accurate but might be slower
    model_name = "Facenet512"  # More accurate version of Facenet
    # model_name = "DeepFace"    # Another option to try
    
    setup_face_recognition(model_name)
    run_webcam_recognition(reference_dir, model_name)

if __name__ == "__main__":
    main()