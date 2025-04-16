import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import face_recognition
import cv2
import os
from datetime import datetime
import yagmail
import geocoder
from PIL import Image

# === CONFIGURATION ===
EMAIL_SENDER = "arkjakki@gmail.com"
EMAIL_PASSWORD = "kraq mohd igao kiyd"
EMAIL_RECEIVER = "arkjakki@gmail.com"

# === Load Known Faces ===
@st.cache_resource
def load_known_faces():
    known_encodings = []
    known_names = []
    dataset_path = 'dataset'

    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path)

            if img is None:
                continue

            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0].split('(')[0].strip().upper()
                known_names.append(name)

    return known_encodings, known_names

known_encodings, known_names = load_known_faces()
notified_names = set()


# === Face Recognition Transformer for Streamlit ===
class FaceRecognitionTransformer(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb_image, (0, 0), fx=0.25, fy=0.25)

        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match = face_distances.argmin() if len(face_distances) > 0 else None

            if best_match is not None and matches[best_match]:
                name = known_names[best_match]

                if name not in notified_names:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    g = geocoder.ip('me')
                    location = g.city + ", " + g.country if g.ok else "Unknown"

                    # Save snapshot
                    snapshot_name = f"{name}_{datetime.now().strftime('%H%M%S')}.jpg"
                    cv2.imwrite(snapshot_name, image)

                    try:
                        yag = yagmail.SMTP(EMAIL_SENDER, EMAIL_PASSWORD)
                        yag.send(
                            to=EMAIL_RECEIVER,
                            subject=f"Missing Person Found: {name}",
                            contents=[
                                f"Name: {name}",
                                f"Time: {now}",
                                f"Location: {location}",
                                "Detected via Streamlit webcam.",
                                snapshot_name
                            ]
                        )
                        notified_names.add(name)
                    except Exception as e:
                        print(f"Email Error: {e}")

                # Draw green box only for known face
                top, right, bottom, left = [v * 4 for v in face_location]
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return image


# === Streamlit UI ===
st.title("📷 Missing Person Detector (Real-Time)")
st.markdown("✅ Check the box below to activate your webcam and start face recognition.")

start = st.checkbox("Start Webcam")

if start:
    webrtc_streamer(
        key="face-detection",
        video_transformer_factory=FaceRecognitionTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
