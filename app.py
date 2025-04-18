import cv2
import face_recognition
import pickle
import serial
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime
import numpy as np

# Load face encodings
with open("encoded_faces/encodings.pkl", "rb") as f:
    data = pickle.load(f)

# Directories
output_dir = "snapshots"
unknown_dir = "unknown_faces"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(unknown_dir, exist_ok=True)

# Arduino setup
# arduino = serial.Serial('COM8', 9600)
# last_sent = ""

# Email config (change as needed)
EMAIL_SENDER = "arkjakki@gmail.com"
EMAIL_PASSWORD = "kraq mohd igao kiyd"
EMAIL_RECEIVER = "arkjakki@example.com"
email_sent_faces = set()

# Helper: send email
def send_email_alert(subject, body, attachment_path):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    if attachment_path:
        with open(attachment_path, 'rb') as file:
            part = MIMEApplication(file.read(), Name=os.path.basename(attachment_path))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
            msg.attach(part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print(f"[EMAIL] Alert sent to {EMAIL_RECEIVER}")
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")

# Start camera
cap = cv2.VideoCapture(0)
label_buffer = []

# Constants
FACE_MATCH_THRESHOLD = 0.5
UNKNOWN_REPEAT_LIMIT = 5
unknown_counter = {}

print("[INFO] Starting video stream...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, face_locations)

    for (top, right, bottom, left), encoding in zip(face_locations, encodings):
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=FACE_MATCH_THRESHOLD)
        face_distances = face_recognition.face_distance(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            best_match_index = np.argmin(face_distances)
            confidence = round((1 - face_distances[best_match_index]) * 100, 2)
            matched_name = data["names"][best_match_index]
            name = f"{matched_name})"

            label_buffer.append(name)
            if len(label_buffer) > 5:
                label_buffer.pop(0)

            # Majority voting buffer
            if label_buffer.count(name) > 3 and last_sent != 'A':
                arduino.write(b'A')
                last_sent = 'A'
                print(f"[ARDUINO] Sent 'A' for {matched_name}")

                filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{matched_name}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, frame)

        else:
            name = "Unknown"
            label_buffer.append(name)
            if len(label_buffer) > 5:
                label_buffer.pop(0)

            # Convert encoding to tuple key for uniqueness
            key = tuple(np.round(encoding, 4))
            unknown_counter[key] = unknown_counter.get(key, 0) + 1

            if unknown_counter[key] == UNKNOWN_REPEAT_LIMIT and last_sent != 'O':
                filename = f"Unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                filepath = os.path.join(unknown_dir, filename)
                cv2.imwrite(filepath, frame)

                arduino.write(b'O')
                last_sent = 'O'
                print("[ARDUINO] Sent 'O' for unknown")

                # Send email once per unknown
                if key not in email_sent_faces:
                    subject = "Alert: Unknown Face Detected"
                    body = f"An unknown person was detected on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
                    send_email_alert(subject, body, filepath)
                    email_sent_faces.add(key)

        # Show result
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
# arduino.close()
print("[INFO] Shutdown complete.")
