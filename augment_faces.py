import os
import cv2
import numpy as np
import random
import face_recognition

def apply_augmentations(image):
    aug_list = []

    # Original
    aug_list.append(image)

    # Flip
    aug_list.append(cv2.flip(image, 1))

    # Bright/Dark
    aug_list.append(cv2.convertScaleAbs(image, alpha=1.2, beta=30))
    aug_list.append(cv2.convertScaleAbs(image, alpha=0.8, beta=-30))

    # Blur
    aug_list.append(cv2.GaussianBlur(image, (5, 5), 0))

    # Rotate
    rows, cols, _ = image.shape
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        aug_list.append(rotated)

    # Noise
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    aug_list.append(cv2.add(image, noise))

    return aug_list

def augment_faces(input_dir="dataset", output_dir="augmented_dataset", crop_faces=True):
    os.makedirs(output_dir, exist_ok=True)

    for person in os.listdir(input_dir):
        in_folder = os.path.join(input_dir, person)
        out_folder = os.path.join(output_dir, person)
        os.makedirs(out_folder, exist_ok=True)

        for img_name in os.listdir(in_folder):
            img_path = os.path.join(in_folder, img_name)
            image = cv2.imread(img_path)

            if crop_faces:
                # Crop only faces
                face_locations = face_recognition.face_locations(image)
                for i, loc in enumerate(face_locations):
                    top, right, bottom, left = loc
                    face_crop = image[top:bottom, left:right]
                    augmented = apply_augmentations(face_crop)
                    for j, aug in enumerate(augmented):
                        aug_name = f"{os.path.splitext(img_name)[0]}_face{i}_aug{j}.jpg"
                        cv2.imwrite(os.path.join(out_folder, aug_name), aug)
            else:
                # Full image
                augmented = apply_augmentations(image)
                for j, aug in enumerate(augmented):
                    aug_name = f"{os.path.splitext(img_name)[0]}_aug{j}.jpg"
                    cv2.imwrite(os.path.join(out_folder, aug_name), aug)

    print(f"[INFO] Augmented dataset saved to {output_dir}")

if __name__ == "__main__":
    augment_faces()
