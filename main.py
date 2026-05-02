import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
import threading
from gestures import detect_gesture, ALL_GESTURES

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Model indiriliyor...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model indirildi.")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

latest_landmarks = []
lock = threading.Lock()

def on_result(result, output_image, timestamp_ms):
    with lock:
        global latest_landmarks
        latest_landmarks = result.hand_landmarks if result.hand_landmarks else []

def draw_landmarks(frame, hand_landmarks_list):
    h, w = frame.shape[:2]
    for hand_landmarks in hand_landmarks_list:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=on_result
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Hand Capture", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Hand Capture", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

fps_time = time.time()
fps = 0

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        small = cv2.resize(frame, (320, 240))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        with lock:
            current_landmarks = list(latest_landmarks)

        # landmark koordinatlarını orijinal frame boyutuna ölçekle
        if current_landmarks:
            h, w = frame.shape[:2]
            for i, hand_landmarks in enumerate(current_landmarks):
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
                for point in points:
                    cv2.circle(frame, point, 4, (0, 0, 255), -1)

                gesture = detect_gesture(hand_landmarks)
                if gesture:
                    x, y = points[0]
                    cv2.putText(frame, gesture, (x - 30, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        fps = 1 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        h, w = frame.shape[:2]
        with lock:
            current_for_panel = list(latest_landmarks)

        detected = []
        for hand_landmarks in current_for_panel:
            g = detect_gesture(hand_landmarks)
            if g:
                detected.append(g)
        active = detected[0] if detected else None

        # Sag alt: aktif hareket
        panel_text = active if active else "No hand"
        (tw, th), _ = cv2.getTextSize(panel_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        pad = 12
        rx1, ry1 = w - tw - pad * 2 - 10, h - th - pad * 2 - 10
        rx2, ry2 = w - 10, h - 10
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 0), -1)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 1)
        cv2.putText(frame, panel_text, (rx1 + pad, ry2 - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Ust sol: gesture listesi
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_h = 28
        list_x = 10
        list_y_start = 60
        label = "Gestures:"
        cv2.putText(frame, label, (list_x, list_y_start), font, 0.6, (200, 200, 200), 1)
        for idx, name in enumerate(ALL_GESTURES):
            y = list_y_start + (idx + 1) * line_h
            color = (0, 255, 0) if name == active else (180, 180, 180)
            prefix = "> " if name == active else "  "
            cv2.putText(frame, prefix + name, (list_x, y), font, 0.65, color, 1 if name != active else 2)

        cv2.imshow("Hand Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
