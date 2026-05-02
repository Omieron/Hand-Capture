import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
import threading
from gestures import smooth_gesture, ALL_GESTURES
from spells import SpellEngine, SPELLS, ABBREV, get_spell_progress

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

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

engine          = SpellEngine()
active_spell    = None
activation_time = 0.0
fps_time        = time.time()
FONT            = cv2.FONT_HERSHEY_SIMPLEX


# ─── Draw helpers ──────────────────────────────────────────────────────────────

def draw_buffer(frame, buffer, hold_gest, hold_prog, cooling_down):
    h, w   = frame.shape[:2]
    box_w  = 88
    box_h  = 34
    gap    = 6
    shown  = buffer[-9:]
    count  = len(shown) + (1 if hold_gest else 0)
    if count == 0 and not cooling_down:
        return

    total_w = max(count, 1) * (box_w + gap) - gap
    sx = (w - total_w) // 2
    sy = h - box_h - 18

    fail_color = (0, 0, 200)

    for i, g in enumerate(shown):
        x     = sx + i * (box_w + gap)
        color = fail_color if cooling_down else (0, 170, 0)
        cv2.rectangle(frame, (x, sy), (x + box_w, sy + box_h), color, -1)
        label = ABBREV.get(g, g[:4])
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.putText(frame, label, (x + (box_w - tw) // 2, sy + 23), FONT, 0.5, (0, 0, 0), 1)

    if hold_gest and not cooling_down:
        x = sx + len(shown) * (box_w + gap)
        cv2.rectangle(frame, (x, sy), (x + box_w, sy + box_h), (50, 50, 50), -1)
        fill = int(box_w * hold_prog)
        if fill > 0:
            cv2.rectangle(frame, (x, sy), (x + fill, sy + box_h), (0, 130, 255), -1)
        cv2.rectangle(frame, (x, sy), (x + box_w, sy + box_h), (140, 140, 140), 1)
        label = ABBREV.get(hold_gest, hold_gest[:4])
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.putText(frame, label, (x + (box_w - tw) // 2, sy + 23), FONT, 0.5, (255, 255, 255), 1)

    if cooling_down:
        remaining = engine.cooldown_remaining
        cd_text   = f"COOLDOWN  {remaining:.1f}s"
        (tw, th), _ = cv2.getTextSize(cd_text, FONT, 0.75, 2)
        cv2.putText(frame, cd_text, ((w - tw) // 2, sy - 10), FONT, 0.75, (0, 0, 220), 2)


SHORT = {
    "Fist": "Fist", "Open Hand": "Open", "Thumbs Up": "T.Up",
    "Point": "Point", "Peace": "Peace", "Rock": "Rock",
    "Four": "Four", "OK": "OK",
}

def draw_spell_list(frame, buffer):
    h, w = frame.shape[:2]
    x0   = w - 358
    y    = 55

    # Arka plan
    panel_h = len(SPELLS) * 56 + 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 8, 40), (w - 5, 40 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    cv2.putText(frame, "Spells:", (x0, y), FONT, 0.6, (200, 200, 200), 1)
    y += 22

    for spell in SPELLS:
        seq  = spell["sequence"]
        prog = get_spell_progress(buffer, seq)
        n    = len(seq)

        # Spell adı + ilerleme sayacı
        label      = f"{spell['name']}   {prog}/{n}"
        name_color = (0, 255, 120) if prog > 0 else (170, 170, 170)
        cv2.putText(frame, label, (x0, y), FONT, 0.52, name_color, 2 if prog > 0 else 1)
        y += 20

        # Gesture isimleri yan yana, eşleşenler yeşil, bekleyenler gri
        # 5'ten uzunsa iki satıra böl
        lines      = [seq[:5], seq[5:]] if len(seq) > 5 else [seq]
        idx_offset = 0
        for line in lines:
            if not line:
                continue
            xpos = x0
            for j, g in enumerate(line):
                gi    = idx_offset + j
                short = SHORT.get(g, g[:5])
                color = (0, 210, 80) if gi < prog else (120, 120, 120)
                cv2.putText(frame, short, (xpos, y), FONT, 0.38, color, 1)
                (tw, _), _ = cv2.getTextSize(short, FONT, 0.38, 1)
                xpos += tw + 3
                if j < len(line) - 1:
                    cv2.putText(frame, "->", (xpos, y), FONT, 0.35, (60, 60, 60), 1)
                    (aw, _), _ = cv2.getTextSize("->", FONT, 0.35, 1)
                    xpos += aw + 3
            idx_offset += len(line)
            y += 15

        y += 6


def draw_activation(frame, spell, elapsed):
    if elapsed > 2.0:
        return
    h, w   = frame.shape[:2]
    alpha  = max(0.0, 1.0 - elapsed / 2.0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), spell["color"], -1)
    cv2.addWeighted(overlay, alpha * 0.35, frame, 1 - alpha * 0.35, 0, frame)

    text  = spell["name"].upper() + "!"
    scale = 3.0
    thick = 4
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    x = (w - tw) // 2
    y = h // 2 + th // 2
    cv2.putText(frame, text, (x + 3, y + 3), FONT, scale, (0, 0, 0), thick + 2)
    cv2.putText(frame, text, (x, y),          FONT, scale, (255, 255, 255), thick)


# ─── Main loop ─────────────────────────────────────────────────────────────────

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        small = cv2.resize(frame, (320, 240))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        with lock:
            current_landmarks = list(latest_landmarks)

        h, w   = frame.shape[:2]
        active = None

        for hand_landmarks in current_landmarks:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            for s, e in HAND_CONNECTIONS:
                cv2.line(frame, points[s], points[e], (0, 255, 0), 2)
            for pt in points:
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)
            g = smooth_gesture(hand_landmarks)
            if g and active is None:
                active = g
                x0, y0 = points[0]
                cv2.putText(frame, g, (x0 - 30, y0 - 20), FONT, 1, (255, 255, 255), 2)

        # Spell engine
        result = engine.update(active or "")
        if result and result != "FAILED":
            active_spell    = result
            activation_time = time.time()

        # ── UI ──

        # Sol üst: gesture listesi
        cv2.putText(frame, "Gestures:", (10, 60), FONT, 0.6, (200, 200, 200), 1)
        for idx, name in enumerate(ALL_GESTURES):
            gy     = 60 + (idx + 1) * 28
            color  = (0, 255, 0) if name == active else (180, 180, 180)
            prefix = "> " if name == active else "  "
            cv2.putText(frame, prefix + name, (10, gy), FONT,
                        0.65, color, 1 if name != active else 2)

        # Sag: spell listesi
        draw_spell_list(frame, engine.buffer)

        # Sag alt: aktif gesture
        panel_text = active if active else "No hand"
        (tw, th), _ = cv2.getTextSize(panel_text, FONT, 1, 2)
        pad = 12
        px1, py1 = w - tw - pad*2 - 10, h - th - pad*2 - 10
        cv2.rectangle(frame, (px1, py1), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (px1, py1), (w - 10, h - 10), (255, 255, 255), 1)
        cv2.putText(frame, panel_text, (px1 + pad, h - pad - 10), FONT, 1, (0, 255, 255), 2)

        # Alt orta: buffer + hold progress
        draw_buffer(frame, engine.buffer, engine.hold_gesture,
                    engine.hold_progress, engine.is_cooling_down)

        # Büyü aktivasyon efekti
        if active_spell:
            draw_activation(frame, active_spell, time.time() - activation_time)

        # FPS
        fps = 1 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), FONT, 1, (255, 255, 0), 2)

        cv2.imshow("Hand Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
