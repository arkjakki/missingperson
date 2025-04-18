import os
import cv2
import numpy as np
import face_recognition
import pickle
from imgaug import augmenters as iaa
from PIL import Image

# Augment image and save
def augment_image(image_path, save_dir):
    image = cv2.imread(image_path)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), 
        iaa.Affine(rotate=(-20, 20)),
        iaa.Multiply((0.8, 1.2)), 
    ])

    for i in range(5):
        augmented = seq(image=image)
        filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_aug_{i}.jpg"
        cv2.imwrite(os.path.join(save_dir, filename), augmented)

# Encode and update pickle
def update_encodings(name, folder_path, encoding_file="encoded_faces/encodings.pkl"):
    known_encodings = []
    known_names = []

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = face_recognition.load_image_file(image_path)
        encs = face_recognition.face_encodings(image)
        if encs:
            known_encodings.append(encs[0])
            known_names.append(name)

    # Load existing if exists
    if os.path.exists(encoding_file):
        with open(encoding_file, "rb") as f:
            data = pickle.load(f)
        known_encodings = data["encodings"] + known_encodings
        known_names = data["names"] + known_names

    # Save updated encodings
    with open(encoding_file, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)
