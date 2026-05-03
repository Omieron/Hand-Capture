import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import math
import os
import time
import threading
from gestures import ALL_GESTURES

MODEL_PATH = "hand_landmarker.task"
CSV_PATH   = "gesture_data.csv"
TARGET     = 50
FONT       = cv2.FONT_HERSHEY_SIMPLEX

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

latest_landmarks = []
lock = threading.Lock()

def on_result(result, output_image, timestamp_ms):
    with lock:
        global latest_landmarks
        latest_landmarks = result.hand_landmarks if result.hand_landmarks else []

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=on_result
)

def extract_features(landmarks):
    wx, wy = landmarks[0].x, landmarks[0].y
    pts = [(lm.x - wx, lm.y - wy) for lm in landmarks]
    scale = math.hypot(pts[9][0], pts[9][1])
    if scale < 1e-6:
        return None
    pts = [(x / scale, y / scale) for x, y in pts]
    return [v for pt in pts for v in pt]

# Mevcut veriyi yükle
counts = {g: 0 for g in ALL_GESTURES}
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row and row[0] in counts:
                counts[row[0]] += 1


# ── Çizim yardımcıları ────────────────────────────────────────────────────────

def semi_rect(frame, x1, y1, x2, y2, color=(15, 15, 15), alpha=0.70):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def progress_bar(frame, x, y, w, h, ratio, fill_col, bg_col=(40, 40, 40)):
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg_col, -1)
    fill = int(w * min(max(ratio, 0.0), 1.0))
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), fill_col, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 70, 70), 1)


def draw_top_bar(frame, hand_detected):
    h, w = frame.shape[:2]
    semi_rect(frame, 0, 0, w, 42, color=(10, 10, 10), alpha=0.80)
    cv2.line(frame, (0, 42), (w, 42), (55, 55, 55), 1)

    cv2.putText(frame, "GESTURE DATA COLLECTOR", (14, 28), FONT, 0.72, (190, 190, 190), 1)

    if hand_detected:
        hand_txt, hand_col = "  EL ALGILANDI  ", (0, 180, 50)
    else:
        hand_txt, hand_col = "  EL YOK  ", (0, 50, 200)

    (tw, _), _ = cv2.getTextSize(hand_txt, FONT, 0.55, 1)
    bx = w - tw - 20
    cv2.rectangle(frame, (bx - 4, 10), (bx + tw + 4, 34), hand_col, -1)
    cv2.putText(frame, hand_txt, (bx, 28), FONT, 0.55, (255, 255, 255), 1)


def draw_gesture_list(frame, counts, current_idx):
    h, w   = frame.shape[:2]
    px, pw = 12, 240
    py     = 50
    row_h  = 54
    ph     = len(ALL_GESTURES) * row_h + 44

    semi_rect(frame, px, py, px + pw, py + ph, color=(12, 12, 12), alpha=0.75)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (55, 55, 55), 1)

    cv2.putText(frame, "GESTURE LIST", (px + 10, py + 24), FONT, 0.58, (160, 160, 160), 1)
    cv2.line(frame, (px + 8, py + 32), (px + pw - 8, py + 32), (55, 55, 55), 1)

    for i, g in enumerate(ALL_GESTURES):
        gy     = py + 44 + i * row_h
        n      = counts[g]
        done   = n >= TARGET
        active = i == current_idx

        # Aktif satır vurgusu
        if active:
            cv2.rectangle(frame, (px + 4, gy - 2), (px + pw - 4, gy + row_h - 6), (0, 55, 90), -1)
            cv2.rectangle(frame, (px + 4, gy - 2), (px + pw - 4, gy + row_h - 6), (0, 150, 210), 1)

        # Gesture adı
        name_col = (0, 210, 255) if active else ((0, 190, 70) if done else (170, 170, 170))
        badge    = ">" if active else ("v" if done else " ")
        cv2.putText(frame, f"{badge} {g}", (px + 12, gy + 18), FONT,
                    0.56, name_col, 2 if active else 1)

        # Progress bar
        bar_x, bar_y, bar_w, bar_h = px + 12, gy + 26, pw - 24, 9
        fill_col = (0, 210, 255) if active else ((0, 170, 55) if done else (50, 90, 50))
        progress_bar(frame, bar_x, bar_y, bar_w, bar_h, n / TARGET, fill_col)

        # Sayı
        cnt_txt = f"{n}/{TARGET}"
        (tw, _), _ = cv2.getTextSize(cnt_txt, FONT, 0.40, 1)
        cnt_col = (0, 210, 255) if active else ((0, 190, 70) if done else (130, 130, 130))
        cv2.putText(frame, cnt_txt, (px + pw - tw - 10, gy + row_h - 8),
                    FONT, 0.40, cnt_col, 1)


def draw_instruction_card(frame, gesture_name, n, hand_detected):
    h, w   = frame.shape[:2]
    done   = n >= TARGET
    card_w = 500
    card_h = 148
    cx     = (w - card_w) // 2
    cy     = h - card_h - 16

    # Arka plan
    bg_col = (8, 8, 8)
    semi_rect(frame, cx, cy, cx + card_w, cy + card_h, color=bg_col, alpha=0.85)

    # Çerçeve rengi: el algılanmadıysa kırmızı, bitti ise yeşil, normal ise cyan
    if not hand_detected:
        border_col = (0, 50, 200)
    elif done:
        border_col = (0, 180, 50)
    else:
        border_col = (0, 160, 220)
    cv2.rectangle(frame, (cx, cy), (cx + card_w, cy + card_h), border_col, 2)

    # Üst etiket
    label = "TAMAMLANDI!" if done else ("ONCELIKLE ELINI GOSTER" if not hand_detected else "BU GESTU RE YAP, SONRA [R] YE BAS:")
    lbl_col = (0, 180, 50) if done else ((0, 60, 200) if not hand_detected else (140, 140, 140))
    cv2.putText(frame, label, (cx + 16, cy + 22), FONT, 0.48, lbl_col, 1)

    # Büyük gesture adı
    scale, thick = 1.9, 3
    (tw, th), _ = cv2.getTextSize(gesture_name.upper(), FONT, scale, thick)
    gx = cx + 16
    gy = cy + 22 + th + 14
    txt_col = (0, 180, 50) if done else ((80, 80, 200) if not hand_detected else (0, 210, 255))
    cv2.putText(frame, gesture_name.upper(), (gx + 2, gy + 2), FONT, scale, (0, 0, 0), thick + 2)
    cv2.putText(frame, gesture_name.upper(), (gx, gy),          FONT, scale, txt_col,  thick)

    # Progress bar
    pb_x  = cx + 16
    pb_y  = cy + card_h - 40
    pb_w  = card_w - 32
    pb_h  = 16
    pb_col = (0, 180, 50) if done else (0, 180, 220)
    progress_bar(frame, pb_x, pb_y, pb_w, pb_h, n / TARGET, pb_col)

    # Alt yazı
    if done:
        hint = f"{n}/{TARGET}   Tamamlandi!  Sonraki icin [D]"
    else:
        hint = f"{n}/{TARGET}   [R]=Kaydet   [A/D]=Gesture degistir   [Q]=Cik"
    cv2.putText(frame, hint, (pb_x, cy + card_h - 10), FONT, 0.42, (130, 130, 130), 1)


# ── Ana döngü ──────────────────────────────────────────────────────────────────

current_idx = 0
flash_msg   = ""
flash_until = 0.0
snap_alpha  = 0.0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Collect", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Collect", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        small = cv2.resize(frame, (320, 240))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        with lock:
            lm_copy = list(latest_landmarks)

        h, w          = frame.shape[:2]
        hand_detected = bool(lm_copy)

        if lm_copy:
            hand   = lm_copy[0]
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for s, e in HAND_CONNECTIONS:
                cv2.line(frame, points[s], points[e], (0, 255, 0), 2)
            for pt in points:
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)

        gesture_name = ALL_GESTURES[current_idx]
        n            = counts[gesture_name]

        # ── UI katmanları ──
        draw_top_bar(frame, hand_detected)
        draw_gesture_list(frame, counts, current_idx)
        draw_instruction_card(frame, gesture_name, n, hand_detected)

        # Orta bildirim (kısa süreli)
        if time.time() < flash_until:
            (fw, _), _ = cv2.getTextSize(flash_msg, FONT, 1.1, 2)
            fx = (w - fw) // 2
            fy = h // 2
            cv2.rectangle(frame, (fx - 12, fy - 36), (fx + fw + 12, fy + 12), (0, 0, 0), -1)
            cv2.putText(frame, flash_msg, (fx, fy), FONT, 1.1, (0, 230, 255), 2)

        # Fotoğraf flaşı
        if snap_alpha > 0.0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (255, 255, 255), -1)
            cv2.addWeighted(overlay, snap_alpha, frame, 1 - snap_alpha, 0, frame)
            snap_alpha = max(0.0, snap_alpha - 0.10)

        cv2.imshow("Collect", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('d'):
            current_idx = (current_idx + 1) % len(ALL_GESTURES)
        elif key == ord('a'):
            current_idx = (current_idx - 1) % len(ALL_GESTURES)
        elif key == ord('r'):
            if not hand_detected:
                flash_msg   = "Once elini goster!"
                flash_until = time.time() + 1.2
            else:
                features = extract_features(lm_copy[0])
                if features:
                    write_header = not os.path.exists(CSV_PATH)
                    with open(CSV_PATH, "a", newline="") as f:
                        writer = csv.writer(f)
                        if write_header:
                            writer.writerow(["label"] + [f"f{i}" for i in range(42)])
                        writer.writerow([gesture_name] + features)
                    counts[gesture_name] += 1
                    flash_msg   = f"Kaydedildi!  {counts[gesture_name]}/{TARGET}"
                    flash_until = time.time() + 0.7
                    snap_alpha  = 1.0

cap.release()
cv2.destroyAllWindows()
print("\nToplanan ornekler:")
for g, n in counts.items():
    status = "TAMAM" if n >= TARGET else f"EKSIK  ({TARGET - n} tane daha)"
    print(f"  {g:12s}  {n:3d}/{TARGET}  ->  {status}")
