import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import shutil


def augment_and_preprocess_dataset(input_dir, output_dir, required_size=(224, 224)):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

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

                enhancer = ImageEnhance.Brightness(face_pil)
                bright_img = enhancer.enhance(1.3)
                bright_img.save(os.path.join(output_dir, f"{base_name}_face_{i}_bright.jpg"))
                dark_img = enhancer.enhance(0.7)
                dark_img.save(os.path.join(output_dir, f"{base_name}_face_{i}_dark.jpg"))

                cv_face = cv2.cvtColor(np.array(face_pil), cv2.COLOR_RGB2BGR)

                for angle in [-15, 15]:
                    matrix = cv2.getRotationMatrix2D((cv_face.shape[1]/2, cv_face.shape[0]/2), angle, 1.0)
                    rotated = cv2.warpAffine(cv_face, matrix, (cv_face.shape[1], cv_face.shape[0]))
                    cv2.imwrite(os.path.join(output_dir, f"{base_name}_face_{i}_rot{angle}.jpg"), rotated)

                zoom_factor = 1.1
                zoomed = cv2.resize(cv_face, None, fx=zoom_factor, fy=zoom_factor)
                start_y = (zoomed.shape[0] - cv_face.shape[0]) // 2
                start_x = (zoomed.shape[1] - cv_face.shape[1]) // 2
                zoomed = zoomed[start_y:start_y+cv_face.shape[0], start_x:start_x+cv_face.shape[1]]
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_face_{i}_zoomed.jpg"), zoomed)

    print(f"Processing and augmentation completed. Images saved in {output_dir}")


def main():
    input_dir = "dataset/iryna"
    output_dir = "dataset_preprocessed/iryna"

    augment_and_preprocess_dataset(input_dir, output_dir)

if __name__ == "__main__":
    main()