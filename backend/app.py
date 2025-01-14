#backend/app.py

import os
import cv2
import time
from flask import Flask, request, jsonify
from deepface import DeepFace
import numpy as np

app = Flask(__name__)

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset')
MODELS_PATH = os.path.join(os.path.dirname(__file__), 'models')
MODEL_FILE = os.path.join(MODELS_PATH, 'face_model.pkl')

# Helper function to capture images via webcam
def capture_images(name, num_samples=5):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False

    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Capture', frame)
        img_name = f"{name}_{time.time()}.jpg"
        cv2.imwrite(os.path.join(DATASET_PATH, img_name), frame)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return True

# Helper function to train the model using DeepFace
def train_model():
    # DeepFace uses database directory approach; you can adapt as needed
    try:
        DeepFace.find("dummy.jpg", db_path=DATASET_PATH)  # Creates representations
        return True
    except Exception as e:
        print(e)
        return False

# Helper function to recognize faces
def recognize_face(image_path):
    try:
        df = DeepFace.find(img_path=image_path, db_path=DATASET_PATH, enforce_detection=False)
        if len(df) > 0:
            # Get best match
            best_match = df.iloc[0].identity
            name = os.path.basename(best_match).split('_')[0]
            return name
        else:
            return "unrecognized"
    except:
        return "unrecognized"

@app.route('/build-dataset', methods=['POST'])
def build_dataset():
    name = request.form.get('name', 'Unknown')
    succeeded = capture_images(name=name)
    if succeeded:
        return jsonify({"status": "success", "message": "Images captured."})
    return jsonify({"status": "error", "message": "Failed to capture images."})

@app.route('/train-model', methods=['GET'])
def train():
    result = train_model()
    if result:
        return jsonify({"status": "success", "message": "Model trained."})
    return jsonify({"status": "error", "message": "Training failed."})

@app.route('/recognize', methods=['POST'])
def recognize():
    file = request.files['image']
    if file:
        img_path = os.path.join(MODELS_PATH, file.filename)
        file.save(img_path)
        recognized_name = recognize_face(img_path)
        return jsonify({"name": recognized_name})
    return jsonify({"name": "unrecognized"})

if __name__ == '__main__':
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    app.run(host='0.0.0.0', port=5000, debug=True)