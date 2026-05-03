import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import csv
import math
import os
import time
import threading
from gestures import smooth_gesture, ALL_GESTURES
from spells import SpellEngine, SPELLS, ABBREV, get_spell_progress

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
CSV_PATH   = "gesture_data.csv"
TARGET     = 50
FONT       = cv2.FONT_HERSHEY_SIMPLEX

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


# ── Collect mode yardımcıları ─────────────────────────────────────────────────

def extract_features(landmarks):
    wx, wy = landmarks[0].x, landmarks[0].y
    pts    = [(lm.x - wx, lm.y - wy) for lm in landmarks]
    scale  = math.hypot(pts[9][0], pts[9][1])
    if scale < 1e-6:
        return None
    pts = [(x / scale, y / scale) for x, y in pts]
    return [v for pt in pts for v in pt]

def load_counts():
    counts = {g: 0 for g in ALL_GESTURES}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and row[0] in counts:
                    counts[row[0]] += 1
    return counts

def semi_rect(frame, x1, y1, x2, y2, color=(15, 15, 15), alpha=0.72):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def pbar(frame, x, y, w, h, ratio, fill_col, bg=(40, 40, 40)):
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
    fill = int(w * min(max(ratio, 0.0), 1.0))
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), fill_col, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 70, 70), 1)

def draw_collect_list(frame, counts, current_idx):
    h, w   = frame.shape[:2]
    px, pw = 12, 238
    py     = 50
    row_h  = 54
    ph     = len(ALL_GESTURES) * row_h + 44

    semi_rect(frame, px, py, px + pw, py + ph, color=(10, 10, 10), alpha=0.78)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (55, 55, 55), 1)
    cv2.putText(frame, "GESTURE LIST", (px + 10, py + 24), FONT, 0.58, (155, 155, 155), 1)
    cv2.line(frame, (px + 8, py + 32), (px + pw - 8, py + 32), (55, 55, 55), 1)

    for i, g in enumerate(ALL_GESTURES):
        gy     = py + 44 + i * row_h
        n      = counts[g]
        done   = n >= TARGET
        active = i == current_idx

        if active:
            cv2.rectangle(frame, (px + 4, gy - 2), (px + pw - 4, gy + row_h - 6), (0, 55, 90), -1)
            cv2.rectangle(frame, (px + 4, gy - 2), (px + pw - 4, gy + row_h - 6), (0, 150, 210), 1)

        name_col = (0, 210, 255) if active else ((0, 185, 65) if done else (165, 165, 165))
        badge    = ">" if active else ("v" if done else " ")
        cv2.putText(frame, f"{badge} {g}", (px + 12, gy + 18), FONT,
                    0.56, name_col, 2 if active else 1)

        bar_x, bar_y, bar_w, bar_h = px + 12, gy + 26, pw - 24, 9
        fill_col = (0, 210, 255) if active else ((0, 165, 50) if done else (50, 90, 50))
        pbar(frame, bar_x, bar_y, bar_w, bar_h, n / TARGET, fill_col)

        cnt = f"{n}/{TARGET}"
        (tw, _), _ = cv2.getTextSize(cnt, FONT, 0.40, 1)
        cv2.putText(frame, cnt, (px + pw - tw - 10, gy + row_h - 8),
                    FONT, 0.40, name_col, 1)

def draw_collect_card(frame, gesture_name, n, hand_detected):
    h, w   = frame.shape[:2]
    done   = n >= TARGET
    card_w = 500
    card_h = 150
    cx     = (w - card_w) // 2
    cy     = h - card_h - 16

    semi_rect(frame, cx, cy, cx + card_w, cy + card_h, color=(8, 8, 8), alpha=0.88)

    if not hand_detected:
        border = (0, 50, 200)
    elif done:
        border = (0, 175, 50)
    else:
        border = (0, 155, 215)
    cv2.rectangle(frame, (cx, cy), (cx + card_w, cy + card_h), border, 2)

    if done:
        lbl, lbl_col = "TAMAMLANDI!", (0, 175, 50)
    elif not hand_detected:
        lbl, lbl_col = "ONCE ELINI GOSTER", (0, 55, 200)
    else:
        lbl, lbl_col = "BU GESTURE YAP  >>  [R] ile kaydet", (130, 130, 130)
    cv2.putText(frame, lbl, (cx + 16, cy + 22), FONT, 0.48, lbl_col, 1)

    scale, thick = 1.9, 3
    (tw, th), _ = cv2.getTextSize(gesture_name.upper(), FONT, scale, thick)
    gx = cx + 16
    gy = cy + 22 + th + 14
    txt_col = (0, 175, 50) if done else ((80, 80, 200) if not hand_detected else (0, 210, 255))
    cv2.putText(frame, gesture_name.upper(), (gx + 2, gy + 2), FONT, scale, (0, 0, 0), thick + 2)
    cv2.putText(frame, gesture_name.upper(), (gx, gy),          FONT, scale, txt_col,  thick)

    pb_x, pb_y, pb_w, pb_h = cx + 16, cy + card_h - 40, card_w - 32, 16
    pbar(frame, pb_x, pb_y, pb_w, pb_h, n / TARGET,
         (0, 175, 50) if done else (0, 175, 215))

    if done:
        hint = f"{n}/{TARGET}   Tamamlandi!  Sonraki icin [D]"
    else:
        hint = f"{n}/{TARGET}   [R]=Kaydet   [A/D]=Gesture sec   [C]=Spell moduna don"
    cv2.putText(frame, hint, (pb_x, cy + card_h - 10), FONT, 0.42, (125, 125, 125), 1)

def draw_collect_topbar(frame, hand_detected):
    h, w = frame.shape[:2]
    semi_rect(frame, 0, 0, w, 42, color=(10, 10, 10), alpha=0.82)
    cv2.line(frame, (0, 42), (w, 42), (55, 55, 55), 1)
    cv2.putText(frame, "VERI KAYIT MODU", (14, 28), FONT, 0.72, (0, 210, 255), 2)

    htxt = "EL ALGILANDI" if hand_detected else "EL YOK"
    hcol = (0, 175, 50)  if hand_detected else (0, 50, 200)
    (tw, _), _ = cv2.getTextSize(htxt, FONT, 0.55, 1)
    bx = w - tw - 28
    cv2.rectangle(frame, (bx - 6, 9), (bx + tw + 6, 35), hcol, -1)
    cv2.putText(frame, htxt, (bx, 28), FONT, 0.55, (255, 255, 255), 1)


# ── Spell mode çizim fonksiyonları ────────────────────────────────────────────

SHORT = {
    "Fist": "Fist", "Open Hand": "Open", "Thumbs Up": "T.Up",
    "Point": "Point", "Peace": "Peace", "Rock": "Rock",
    "Four": "Four", "OK": "OK",
}

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

    for i, g in enumerate(shown):
        x     = sx + i * (box_w + gap)
        color = (0, 0, 200) if cooling_down else (0, 170, 0)
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
        cd_text = f"COOLDOWN  {engine.cooldown_remaining:.1f}s"
        (tw, th), _ = cv2.getTextSize(cd_text, FONT, 0.75, 2)
        cv2.putText(frame, cd_text, ((w - tw) // 2, sy - 10), FONT, 0.75, (0, 0, 220), 2)

def draw_spell_list(frame, buffer):
    h, w = frame.shape[:2]
    x0   = w - 358
    y    = 55

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

        label      = f"{spell['name']}   {prog}/{n}"
        name_color = (0, 255, 120) if prog > 0 else (170, 170, 170)
        cv2.putText(frame, label, (x0, y), FONT, 0.52, name_color, 2 if prog > 0 else 1)
        y += 20

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
    scale, thick = 3.0, 4
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    x = (w - tw) // 2
    y = h // 2 + th // 2
    cv2.putText(frame, text, (x + 3, y + 3), FONT, scale, (0, 0, 0), thick + 2)
    cv2.putText(frame, text, (x, y),          FONT, scale, (255, 255, 255), thick)


# ── State ─────────────────────────────────────────────────────────────────────

collect_mode  = False
collect_idx   = 0
counts        = load_counts()
flash_msg     = ""
flash_until   = 0.0
snap_alpha    = 0.0

engine          = SpellEngine()
active_spell    = None
activation_time = 0.0
fps_time        = time.time()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Hand Capture", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Hand Capture", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# ── Ana döngü ─────────────────────────────────────────────────────────────────

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

        h, w          = frame.shape[:2]
        hand_detected = bool(current_landmarks)
        active        = None

        for hand_landmarks in current_landmarks:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            for s, e in HAND_CONNECTIONS:
                cv2.line(frame, points[s], points[e], (0, 255, 0), 2)
            for pt in points:
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)
            g = smooth_gesture(hand_landmarks)
            if g and active is None:
                active = g
                if not collect_mode:
                    x0, y0 = points[0]
                    cv2.putText(frame, g, (x0 - 30, y0 - 20), FONT, 1, (255, 255, 255), 2)

        # ── COLLECT MODE UI ───────────────────────────────────────────────────
        if collect_mode:
            gesture_name = ALL_GESTURES[collect_idx]
            n            = counts[gesture_name]

            draw_collect_topbar(frame, hand_detected)
            draw_collect_list(frame, counts, collect_idx)
            draw_collect_card(frame, gesture_name, n, hand_detected)

            # Kısa bildirim
            if time.time() < flash_until:
                (fw, _), _ = cv2.getTextSize(flash_msg, FONT, 1.1, 2)
                fx, fy     = (w - fw) // 2, h // 2
                cv2.rectangle(frame, (fx - 12, fy - 36), (fx + fw + 12, fy + 12), (0, 0, 0), -1)
                cv2.putText(frame, flash_msg, (fx, fy), FONT, 1.1, (0, 230, 255), 2)

            # Fotoğraf flaşı
            if snap_alpha > 0.0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (255, 255, 255), -1)
                cv2.addWeighted(overlay, snap_alpha, frame, 1 - snap_alpha, 0, frame)
                snap_alpha = max(0.0, snap_alpha - 0.10)

        # ── SPELL MODE UI ─────────────────────────────────────────────────────
        else:
            result = engine.update(active or "")
            if result and result != "FAILED":
                active_spell    = result
                activation_time = time.time()

            cv2.putText(frame, "Gestures:", (10, 60), FONT, 0.6, (200, 200, 200), 1)
            for idx, name in enumerate(ALL_GESTURES):
                gy     = 60 + (idx + 1) * 28
                color  = (0, 255, 0) if name == active else (180, 180, 180)
                prefix = "> " if name == active else "  "
                cv2.putText(frame, prefix + name, (10, gy), FONT,
                            0.65, color, 1 if name != active else 2)

            draw_spell_list(frame, engine.buffer)

            panel_text = active if active else "No hand"
            (tw, th), _ = cv2.getTextSize(panel_text, FONT, 1, 2)
            pad = 12
            px1, py1 = w - tw - pad*2 - 10, h - th - pad*2 - 10
            cv2.rectangle(frame, (px1, py1), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (px1, py1), (w - 10, h - 10), (255, 255, 255), 1)
            cv2.putText(frame, panel_text, (px1 + pad, h - pad - 10), FONT, 1, (0, 255, 255), 2)

            draw_buffer(frame, engine.buffer, engine.hold_gesture,
                        engine.hold_progress, engine.is_cooling_down)

            if active_spell:
                draw_activation(frame, active_spell, time.time() - activation_time)

            # [C] tuşu ipucu
            cv2.putText(frame, "[C] Veri kayit moduna gec", (10, h - 16), FONT, 0.48, (100, 100, 100), 1)

            fps = 1 / (time.time() - fps_time)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), FONT, 1, (255, 255, 0), 2)

        fps_time = time.time()
        cv2.imshow("Hand Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            collect_mode = not collect_mode
            counts       = load_counts()   # modu açınca güncel sayıları yükle

        # Collect mode tuşları
        elif collect_mode:
            if key == ord('d'):
                collect_idx = (collect_idx + 1) % len(ALL_GESTURES)
            elif key == ord('a'):
                collect_idx = (collect_idx - 1) % len(ALL_GESTURES)
            elif key == ord('r'):
                if not hand_detected:
                    flash_msg   = "Once elini goster!"
                    flash_until = time.time() + 1.2
                else:
                    features = extract_features(current_landmarks[0])
                    if features:
                        gesture_name = ALL_GESTURES[collect_idx]
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
