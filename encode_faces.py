import face_recognition
import os
import pickle

DATASET_DIR = "dataset"
ENCODED_DIR = "encoded_faces"

os.makedirs(ENCODED_DIR, exist_ok=True)

known_encodings = []
known_names = []

for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            continue

        encoding = face_recognition.face_encodings(image, face_locations)[0]
        known_encodings.append(encoding)
        known_names.append(person_name)

data = {"encodings": known_encodings, "names": known_names}

with open(os.path.join(ENCODED_DIR, "encodings.pkl"), "wb") as f:
    pickle.dump(data, f)

print(f"[INFO] Encoded {len(known_names)} faces from {len(set(known_names))} people.")
