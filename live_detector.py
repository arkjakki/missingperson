# import cv2
# import face_recognition
# import pickle
# import numpy as np
# import os
# from datetime import datetime
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from email.mime.application import MIMEApplication

# # Load encodings
# with open("encoded_faces/encodings.pkl", "rb") as f:
#     data = pickle.load(f)

# # Output folders
# output_dir = "snapshots"
# unknown_dir = "unknown_faces"
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(unknown_dir, exist_ok=True)

# # Email settings
# EMAIL_SENDER = "arkjakki@gmail.com"
# EMAIL_PASSWORD = "kraq mohd igao kiyd"
# EMAIL_RECEIVER = "arkjakki@example.com"
# email_sent_faces = set()

# # Constants
# FACE_MATCH_THRESHOLD = 0.5
# UNKNOWN_REPEAT_LIMIT = 5
# unknown_counter = {}

# # Email helper
# def send_email_alert(subject, body, attachment_path):
#     msg = MIMEMultipart()
#     msg['From'] = EMAIL_SENDER
#     msg['To'] = EMAIL_RECEIVER
#     msg['Subject'] = subject

#     msg.attach(MIMEText(body, 'plain'))

#     if attachment_path:
#         with open(attachment_path, 'rb') as file:
#             part = MIMEApplication(file.read(), Name=os.path.basename(attachment_path))
#             part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
#             msg.attach(part)

#     try:
#         server = smtplib.SMTP('smtp.gmail.com', 587)
#         server.starttls()
#         server.login(EMAIL_SENDER, EMAIL_PASSWORD)
#         server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
#         server.quit()
#         print(f"[EMAIL] Alert sent to {EMAIL_RECEIVER}")
#     except Exception as e:
#         print(f"[EMAIL ERROR] {e}")

# # Start camera
# cap = cv2.VideoCapture(0)
# label_buffer = []

# print("[INFO] Camera started. Press 'q' to exit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb)
#     encodings = face_recognition.face_encodings(rgb, face_locations)

#     for (top, right, bottom, left), encoding in zip(face_locations, encodings):
#         matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=FACE_MATCH_THRESHOLD)
#         face_distances = face_recognition.face_distance(data["encodings"], encoding)
#         name = "Unknown"

#         if True in matches:
#             best_match_index = np.argmin(face_distances)
#             matched_name = data["names"][best_match_index]
#             name = f"{matched_name}"

#             label_buffer.append(name)
#             if len(label_buffer) > 5:
#                 label_buffer.pop(0)

#             if label_buffer.count(name) > 3:
#                 filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{matched_name}.jpg"
#                 filepath = os.path.join(output_dir, filename)
#                 cv2.imwrite(filepath, frame)

#                 print(f"[MATCH] {matched_name} found and saved at {filepath}")

#         else:
#             name = "Unknown"
#             label_buffer.append(name)
#             if len(label_buffer) > 5:
#                 label_buffer.pop(0)

#             key = tuple(np.round(encoding, 4))
#             unknown_counter[key] = unknown_counter.get(key, 0) + 1

#             if unknown_counter[key] == UNKNOWN_REPEAT_LIMIT:
#                 filename = f"Unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#                 filepath = os.path.join(unknown_dir, filename)
#                 cv2.imwrite(filepath, frame)

#                 print("[ALERT] Unknown person detected and saved.")

#                 if key not in email_sent_faces:
#                     subject = "Alert: Unknown Face Detected"
#                     body = f"Unknown person detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
#                     send_email_alert(subject, body, filepath)
#                     email_sent_faces.add(key)

#         # Draw on screen
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#     cv2.imshow("🔍 Live Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()
# print("[INFO] Detection stopped.")
import cv2
import face_recognition
import pickle
import os
import numpy as np
from datetime import datetime

# Load face encodings
ENCODING_PATH = "encoded_faces/encodings.pkl"
if not os.path.exists(ENCODING_PATH):
    print("[ERROR] No encodings found. Please upload and encode a face first.")
    exit()

with open(ENCODING_PATH, "rb") as f:
    data = pickle.load(f)

# Directories
output_dir = "snapshots"
unknown_dir = "unknown_faces"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(unknown_dir, exist_ok=True)

# Constants
FACE_MATCH_THRESHOLD = 0.5
UNKNOWN_REPEAT_LIMIT = 5
unknown_counter = {}
label_buffer = []

print("[INFO] Starting live camera for real-time face detection...")
cap = cv2.VideoCapture(0)

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
            matched_name = data["names"][best_match_index]
            name = f"{matched_name}"

            # Save snapshot
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{matched_name}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)

        else:
            key = tuple(np.round(encoding, 4))
            unknown_counter[key] = unknown_counter.get(key, 0) + 1

            if unknown_counter[key] == UNKNOWN_REPEAT_LIMIT:
                filename = f"Unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                filepath = os.path.join(unknown_dir, filename)
                cv2.imwrite(filepath, frame)

        # Draw box + name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("🎥 Live Feed - Press Q to Quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Camera closed. Detection ended.")
