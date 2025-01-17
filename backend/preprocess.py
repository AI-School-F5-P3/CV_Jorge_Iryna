import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import shutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(ROOT_DIR, 'backend', 'dataset')
DATASET_PREPROCESSED_PATH = os.path.join(ROOT_DIR, 'backend', 'dataset_preprocessed')
MODELS_PATH = os.path.join(ROOT_DIR, 'backend', 'models')
MODEL_FILE = os.path.join(MODELS_PATH, 'face_model_facenet512.pkl')

def augment_and_preprocess_dataset(name, required_size=(224, 224)):

    input_dir = os.path.join(DATASET_PATH, name)
    output_dir = os.path.join(DATASET_PREPROCESSED_PATH, name)

    # Create output directory if it doesn't exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Ensure input directory exists
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for i, (x, y, w, h) in enumerate(faces):
                face = img[y:y+h, x:x+w]
                base_name = os.path.splitext(filename)[0]
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                face_pil = face_pil.resize(required_size)
                output_path = os.path.join(output_dir, f"{base_name}_face_{i}.jpg")
                face_pil.save(output_path)

                flipped = face_pil.transpose(Image.FLIP_LEFT_RIGHT)
                flipped.save(os.path.join(output_dir, f"{base_name}_face_{i}_flipped.jpg"))

                # Brightness variations
                enhancer = ImageEnhance.Brightness(face_pil)
                bright_img = enhancer.enhance(1.3)
                bright_img.save(os.path.join(output_dir, f"{base_name}_face_{i}_bright.jpg"))
                dark_img = enhancer.enhance(0.7)
                dark_img.save(os.path.join(output_dir, f"{base_name}_face_{i}_dark.jpg"))

                # Rotations and zoom
                cv_face = cv2.cvtColor(np.array(face_pil), cv2.COLOR_RGB2BGR)

                for angle in [-15, 15]:
                    matrix = cv2.getRotationMatrix2D((cv_face.shape[1]/2, cv_face.shape[0]/2), angle, 1.0)
                    rotated = cv2.warpAffine(cv_face, matrix, (cv_face.shape[1], cv_face.shape[0]))
                    cv2.imwrite(os.path.join(output_dir, f"{base_name}_face_{i}_rot{angle}.jpg"), rotated)

                # Zoom effect
                zoom_factor = 1.1
                zoomed = cv2.resize(cv_face, None, fx=zoom_factor, fy=zoom_factor)
                start_y = (zoomed.shape[0] - cv_face.shape[0]) // 2
                start_x = (zoomed.shape[1] - cv_face.shape[1]) // 2
                zoomed = zoomed[start_y:start_y+cv_face.shape[0], start_x:start_x+cv_face.shape[1]]
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_face_{i}_zoomed.jpg"), zoomed)

    print(f"Processing and augmentation completed for {name}. Images saved in {output_dir}")

def main():
    """
    Main function to process all person directories in the dataset folder.
    """
    # Create necessary directories if they don't exist
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(DATASET_PREPROCESSED_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)

    # Process each person's directory
    for person_name in os.listdir(DATASET_PATH):
        person_dir = os.path.join(DATASET_PATH, person_name)
        if os.path.isdir(person_dir):
            print(f"Processing images for {person_name}...")
            try:
                augment_and_preprocess_dataset(person_name)
                print(f"Successfully processed images for {person_name}")
            except Exception as e:
                print(f"Error processing images for {person_name}: {str(e)}")

if __name__ == "__main__":
    main()