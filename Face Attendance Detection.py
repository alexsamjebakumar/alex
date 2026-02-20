import sys
import os
import cv2
import face_recognition
import pandas as pd
from datetime import datetime
import time
import pyttsx3

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout
)
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
known_faces_dir = os.path.join(BASE_DIR, "known_faces")
attendance_file = os.path.join(BASE_DIR, "attendance.csv")

# ---------------- CSV ----------------
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Date", "Session", "Time"]).to_csv(attendance_file, index=False)

# ---------------- VOICE ----------------
engine = pyttsx3.init()
engine.setProperty("rate", 160)
last_alert_time = 0
ALERT_GAP = 5

def unknown_alert():
    global last_alert_time
    now = time.time()
    if now - last_alert_time > ALERT_GAP:
        engine.say("Warning. Unknown person detected")
        engine.runAndWait()
        last_alert_time = now

# ---------------- LOAD FACES ----------------
known_encodings = []
known_names = []

for file in os.listdir(known_faces_dir):
    path = os.path.join(known_faces_dir, file)
    img = face_recognition.load_image_file(path)
    enc = face_recognition.face_encodings(img)
    if enc:
        known_encodings.append(enc[0])
        known_names.append(os.path.splitext(file)[0])

# ---------------- SESSION ----------------
def get_session():
    return "Morning" if datetime.now().hour < 12 else "Evening"

# ---------------- ATTENDANCE ----------------
def mark_attendance(name):
    df = pd.read_csv(attendance_file)
    today = datetime.now().strftime("%Y-%m-%d")
    session = get_session()

    if ((df["Name"] == name) &
        (df["Date"] == today) &
        (df["Session"] == session)).any():
        return

    df.loc[len(df)] = [
        name,
        today,
        session,
        datetime.now().strftime("%H:%M:%S")
    ]
    df.to_csv(attendance_file, index=False)

# ---------------- MAIN WINDOW ----------------
class FaceAttendanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Attendance System")
        self.resize(800, 600)

        self.image_label = QLabel("Camera")
        self.status_label = QLabel("Status : Ready")

        self.start_btn = QPushButton("Start Camera")
        self.stop_btn = QPushButton("Stop Camera")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        self.setLayout(layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.timer.start(30)
        self.status_label.setText("Status : Camera Started")

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.image_label.clear()
        self.status_label.setText("Status : Camera Stopped")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        for (top, right, bottom, left), enc in zip(locations, encodings):
            matches = face_recognition.compare_faces(known_encodings, enc)
            name = "Unknown"

            if True in matches:
                idx = matches.index(True)
                name = known_names[idx]
                mark_attendance(name)
                color = (0, 255, 0)
                self.status_label.setText(f"Attendance saved : {name}")
            else:
                unknown_alert()
                color = (0, 0, 255)
                self.status_label.setText("⚠ Unknown person detected")

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(
                frame,
                f"{name} - {get_session()}",
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
        self.image_label.setPixmap(QPixmap.fromImage(img))

# ---------------- RUN ----------------
app = QApplication(sys.argv)
window = FaceAttendanceApp()
window.show()
sys.exit(app.exec())
