# import streamlit as st
# import os
# import cv2
# import face_recognition
# import numpy as np
# import pickle
# from datetime import datetime
# import shutil

# # Setup
# UPLOAD_FOLDER = "uploads"
# ENCODED_FOLDER = "encoded_faces"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(ENCODED_FOLDER, exist_ok=True)
# PICKLE_FILE = os.path.join(ENCODED_FOLDER, "encodings.pkl")

# # Load previous encodings
# if os.path.exists(PICKLE_FILE):
#     with open(PICKLE_FILE, "rb") as f:
#         data = pickle.load(f)
# else:
#     data = {"encodings": [], "names": []}

# # Streamlit UI
# st.title("🚨 Missing Person Auto Tracker")
# name = st.text_input("Enter Person's Name")
# uploaded_file = st.file_uploader("Upload a clear image", type=["jpg", "jpeg", "png"])

# if uploaded_file and name:
#     file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.read())

#     # Load and detect face
#     image = cv2.imread(file_path)
#     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb)

#     if len(face_locations) != 1:
#         st.error("Please upload an image with exactly ONE clear face.")
#         os.remove(file_path)
#     else:
#         st.success("Face detected. Processing...")

#         # Augmentation (simple version)
#         aug_images = [image]
#         flipped = cv2.flip(image, 1)
#         aug_images.append(flipped)
#         bright = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
#         aug_images.append(bright)

#         for i, aug_img in enumerate(aug_images):
#             rgb_aug = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
#             encoding = face_recognition.face_encodings(rgb_aug)[0]
#             data["encodings"].append(encoding)
#             data["names"].append(name)

#         # Save updated encodings
#         with open(PICKLE_FILE, "wb") as f:
#             pickle.dump(data, f)

#         st.success("Encodings saved and updated!")

#         # Optional: Move file to a permanent folder
#         shutil.move(file_path, os.path.join(ENCODED_FOLDER, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"))

#         # Button to start live detection
#         if st.button("🔍 Start Live Detection"):
#             st.success("Launching live detection...")
#             os.system("python live_detector.py")  # your existing code


import streamlit as st
import os
import cv2
import face_recognition
import numpy as np
import pickle
import serial
import time  # Add this line
from datetime import datetime
import shutil


# Setup
UPLOAD_FOLDER = "uploads"
ENCODED_FOLDER = "encoded_faces"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ENCODED_FOLDER, exist_ok=True)
PICKLE_FILE = os.path.join(ENCODED_FOLDER, "encodings.pkl")

# Load previous encodings
if os.path.exists(PICKLE_FILE):
    with open(PICKLE_FILE, "rb") as f:
        data = pickle.load(f)
else:
    data = {"encodings": [], "names": []}

# Setup Arduino
arduino_port = "COM8"  # Set your Arduino port (COMx)
arduino = serial.Serial(arduino_port, 9600)  # Set baud rate to match Arduino (9600)
time.sleep(2)  # Wait for the serial connection to initialize

# Streamlit UI
st.title("🚨 Missing Person Auto Tracker")
name = st.text_input("Enter Person's Name")
uploaded_file = st.file_uploader("Upload a clear image", type=["jpg", "jpeg", "png"])

if uploaded_file and name:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load and detect face
    image = cv2.imread(file_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)

    if len(face_locations) != 1:
        st.error("Please upload an image with exactly ONE clear face.")
        os.remove(file_path)
    else:
        st.success("Face detected. Processing...")

        # Augmentation (simple version)
        aug_images = [image]
        flipped = cv2.flip(image, 1)
        aug_images.append(flipped)
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
        aug_images.append(bright)

        for i, aug_img in enumerate(aug_images):
            rgb_aug = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
            encoding = face_recognition.face_encodings(rgb_aug)[0]
            data["encodings"].append(encoding)
            data["names"].append(name)

        # Save updated encodings
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(data, f)

        st.success("Encodings saved and updated!")

        # Optional: Move file to a permanent folder
        shutil.move(file_path, os.path.join(ENCODED_FOLDER, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"))

        # Button to start live detection
        if st.button("🔍 Start Live Detection"):
            st.success("Launching live detection...")
            cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                # Flag to check if any known face is detected
                recognized_face = False

                for face_encoding, face_location in zip(face_encodings, face_locations):
                    matches = face_recognition.compare_faces(data["encodings"], face_encoding)
                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = data["names"][first_match_index]
                        recognized_face = True

                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                if recognized_face:
                    # Send signal to Arduino to turn on the LEDs (A: All on)
                    arduino.write(b'A')
                    st.write("Recognized person! LEDs turned ON.")
                else:
                    # Send signal to Arduino to turn off the LEDs (O: All off)
                    arduino.write(b'O')
                    st.write("No recognized person in frame! LEDs turned OFF.")

                # Display the frame
                st.image(frame, channels="BGR")

            cap.release()
            cv2.destroyAllWindows()
