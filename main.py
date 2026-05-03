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

DISP_W, DISP_H = 1280, 720   # gerçek boyut ilk frame'de belirlenir

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
CSV_PATH   = "gesture_data.csv"
TARGET     = 50
FONT       = cv2.FONT_HERSHEY_SIMPLEX

if not os.path.exists(MODEL_PATH):
    print("Model indiriliyor...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

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
    num_hands=2,
    result_callback=on_result
)

# ── Renkler ───────────────────────────────────────────────────────────────────
CYAN   = (0,  210, 255)
GREEN  = (80, 200,  80)
RED    = (55,  50, 215)
ORANGE = (30, 140, 255)
WHITE  = (235, 230, 240)
GREY   = (140, 135, 150)
DIM    = (70,  65,  80)
DARK   = (20,  16,  26)
BORDER = (56,  50,  66)


# ── Çizim araçları ────────────────────────────────────────────────────────────

def rrect(img, x1, y1, x2, y2, r, color):
    x1,y1,x2,y2,r = int(x1),int(y1),int(x2),int(y2),max(1,int(r))
    cv2.rectangle(img, (x1+r,y1), (x2-r,y2), color, -1)
    cv2.rectangle(img, (x1,y1+r), (x2,y2-r), color, -1)
    for cx,cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(img, (cx,cy), r, color, -1)

def panel_bg(ov, x1, y1, x2, y2, r=10):
    """Panelin arka planını overlay'e çizer (tek blend için)."""
    rrect(ov, x1-1, y1-1, x2+1, y2+1, r+1, BORDER)
    rrect(ov, x1,   y1,   x2,   y2,   r,   DARK)

def row_bg(ov, x1, y1, x2, y2, fill, border, r=6):
    rrect(ov, x1-1, y1-1, x2+1, y2+1, r+1, border)
    rrect(ov, x1,   y1,   x2,   y2,   r,   fill)

def lbl(frame, text, x, y, scale=0.55, color=WHITE, bold=False):
    cv2.putText(frame, text, (int(x),int(y)), FONT, scale, color, 2 if bold else 1)

def lbl_c(frame, text, cx, y, scale=0.55, color=WHITE, bold=False):
    t = 2 if bold else 1
    (tw,_),_ = cv2.getTextSize(text, FONT, scale, t)
    cv2.putText(frame, text, (int(cx)-tw//2, int(y)), FONT, scale, color, t)

def pbar(frame, x, y, w, h, ratio, fill_col, r=4):
    rrect(frame, x, y, x+w, y+h, r, (36,32,44))
    fill = int(w * min(max(ratio,0.0),1.0))
    if fill > r*2:
        rrect(frame, x, y, x+fill, y+h, r, fill_col)
    cv2.rectangle(frame, (int(x),int(y)), (int(x+w),int(y+h)), BORDER, 1)

def chip(frame, x, y, text, bg, fg, scale=0.38):
    (tw,th),_ = cv2.getTextSize(text, FONT, scale, 1)
    px,py = 6,3; w,h = tw+px*2, th+py*2; r = h//2
    rrect(frame, x, y, x+w, y+h, r, bg)
    cv2.putText(frame, text, (x+px, y+py+th), FONT, scale, fg, 1)
    return w, h


# ── Collect yardımcıları ──────────────────────────────────────────────────────

def extract_features(landmarks):
    wx,wy = landmarks[0].x, landmarks[0].y
    pts   = [(lm.x-wx, lm.y-wy) for lm in landmarks]
    scale = math.hypot(pts[9][0], pts[9][1])
    if scale < 1e-6: return None
    pts = [(x/scale, y/scale) for x,y in pts]
    return [v for pt in pts for v in pt]

def load_counts():
    counts = {g:0 for g in ALL_GESTURES}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH) as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and row[0] in counts:
                    counts[row[0]] += 1
    return counts


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
fps_val         = 60.0
fps_time        = time.time()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ── Pencere + tam ekran ───────────────────────────────────────────────────────
import numpy as _np

cv2.namedWindow("Hand Capture", cv2.WINDOW_NORMAL)
_blank = _np.zeros((720, 1280, 3), _np.uint8)
cv2.imshow("Hand Capture", _blank)
cv2.waitKey(100)   # pencere oluşsun

# PyObjC ile gerçek borderless fullscreen (macOS title bar + menu bar'ı kaldırır)
DISP_W, DISP_H = 1280, 720
try:
    from AppKit import NSApplication as _NSApp
    import time as _t
    _t.sleep(0.3)   # NSWindow listesinin dolmasını bekle

    for _win in _NSApp.sharedApplication().windows():
        try:
            if "Hand Capture" not in (_win.title() or ""):
                continue
        except Exception:
            continue
        _sf = _win.screen().frame()           # tam ekran frame (menü barı dahil)
        _win.setStyleMask_(0)                 # NSWindowStyleMaskBorderless
        _win.setLevel_(24)                    # NSMainMenuWindowLevel — menü barının üstü
        _win.setFrame_display_(_sf, True)
        _win.makeKeyAndOrderFront_(None)
        DISP_W = int(_sf.size.width)
        DISP_H = int(_sf.size.height)
        break

except Exception as _err:
    # PyObjC çalışmazsa klasik fullscreen'e dön
    try:
        import tkinter as _tk
        _r = _tk.Tk(); _r.withdraw()
        DISP_W, DISP_H = _r.winfo_screenwidth(), _r.winfo_screenheight()
        _r.destroy()
    except Exception:
        pass
    cv2.setWindowProperty("Hand Capture", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Hand Capture", _blank); cv2.waitKey(800)
    _rect = cv2.getWindowImageRect("Hand Capture")
    if _rect and _rect[2] > 100:
        DISP_W, DISP_H = _rect[2], _rect[3]

del _blank


# ── Ana döngü ─────────────────────────────────────────────────────────────────

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        # Kamera frameini işleme çözünürlüğüne getir (1280x720 sabit)
        if frame.shape[1] != 1280:
            frame = cv2.resize(frame, (1280, 720))

        small = cv2.resize(frame, (320,240))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        landmarker.detect_async(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int(time.time()*1000)
        )

        with lock:
            cur_lm = list(latest_landmarks)

        h, w = frame.shape[:2]   # 720, 1280
        hand_detected = bool(cur_lm)
        active    = None
        wrist_pt  = None

        for lms in cur_lm:
            pts = [(int(lm.x*w), int(lm.y*h)) for lm in lms]
            for s,e in HAND_CONNECTIONS:
                cv2.line(frame, pts[s], pts[e], (0,210,0), 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, (40,40,210), -1)
            g = smooth_gesture(lms)
            if g and active is None:
                active   = g
                wrist_pt = pts[0]

        # ── Spell engine güncellemesi ──────────────────────────────────────────
        if not collect_mode:
            result = engine.update(active or "")
            if result and result != "FAILED":
                active_spell    = result
                activation_time = time.time()

        # ══════════════════════════════════════════════════════════════════════
        # RENDER — tüm yarı-saydam arka planlar TEK overlay'e çizilir,
        # sonra tek addWeighted, ardından metin/çizgiler doğrudan frame'e.
        # ══════════════════════════════════════════════════════════════════════
        ov = frame.copy()   # ← tek kopya

        # ── Topbar bg ─────────────────────────────────────────────────────────
        cv2.rectangle(ov, (0,0), (w,44), (13,9,19), -1)

        if collect_mode:
            # ── Collect: sol panel bg ────────────────────────────────────────
            RH  = 54;  PX,PW = 10, 232
            PY  = 52;  PH = len(ALL_GESTURES)*RH + 42
            panel_bg(ov, PX, PY, PX+PW, PY+PH)
            gy_act = PY + 38 + collect_idx * RH
            row_bg(ov, PX+4, gy_act-2, PX+PW-4, gy_act+RH-6, (18,48,36), GREEN)

            # ── Collect: kart bg ──────────────────────────────────────────────
            CW,CH = 490, 148
            CX = (w-CW)//2;  CY = h-CH-14
            panel_bg(ov, CX, CY, CX+CW, CY+CH, r=12)

        else:
            # ── Spell: sağ panel bg ───────────────────────────────────────────
            SPW = 295;  SPX = w-SPW-10;  SPY = 52
            SRH = 64;   SPH = len(SPELLS)*SRH + 38
            panel_bg(ov, SPX, SPY, SPX+SPW, SPY+SPH)
            for i,spell in enumerate(SPELLS):
                if get_spell_progress(engine.buffer, spell["sequence"]) > 0:
                    sy = SPY + 38 + i*SRH
                    row_bg(ov, SPX+4, sy-2, SPX+SPW-4, sy+SRH-6, (32,22,46), DIM)

            # ── Spell: buffer bg (tokenler için zemin) ────────────────────────
            buf   = list(engine.buffer[-8:])
            has_h = bool(engine.hold_gesture)
            tc    = len(buf) + (1 if has_h else 0)
            if tc > 0:
                TW,TH,TG = 80, 34, 6
                tot = tc*(TW+TG)-TG
                bx  = (w-tot)//2;  by = h-TH-14
                cv2.rectangle(ov, (bx-6,by-6), (bx+tot+6,by+TH+6), (14,10,20), -1)

        # ── Tek blend ─────────────────────────────────────────────────────────
        cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)

        # ══════════════════════════════════════════════════════════════════════
        # FOREGROUND — metin, çizgiler, borders (alpha yok, hızlı)
        # ══════════════════════════════════════════════════════════════════════

        # ── Topbar fg ─────────────────────────────────────────────────────────
        cv2.line(frame, (0,44), (w,44), BORDER, 1)
        if collect_mode:
            lbl(frame, "RECORD MODE", 16, 28, 0.70, GREEN, bold=True)
        else:
            lbl(frame, "SPELL CASTER", 16, 28, 0.70, CYAN, bold=True)

        # Kapatma butonu (topbar sağ uç)
        rrect(frame, w-58, 8, w-8, 36, 6, (50,30,30))
        cv2.rectangle(frame, (w-58,8), (w-8,36), (100,55,55), 1)
        lbl_c(frame, "X  ESC", w-33, 27, 0.40, (200,120,120))

        # El durumu (topbar sağ)
        hdot = GREEN if hand_detected else RED
        cv2.circle(frame, (w-78, 22), 6, hdot, -1)
        htxt = "EL VAR" if hand_detected else "EL YOK"
        (htw,_),_ = cv2.getTextSize(htxt, FONT, 0.40, 1)
        lbl(frame, htxt, w-htw-90, 27, 0.40, hdot)

        # Aktif gesture pill (topbar, spell modunda)
        if active and not collect_mode:
            gtxt = ABBREV.get(active, active[:4])
            (gtw,_),_ = cv2.getTextSize(gtxt, FONT, 0.52, 1)
            gx = w - gtw - htw - 70
            rrect(frame, gx-8, 10, gx+gtw+8, 34, 6, (36,26,50))
            cv2.rectangle(frame, (gx-8,10), (gx+gtw+8,34), CYAN, 1)
            lbl(frame, gtxt, gx, 28, 0.52, CYAN)

        # FPS (topbar sağ alt köşe)
        lbl(frame, f"{int(fps_val)}", w-22, 42, 0.36, DIM)

        # ── Elde gesture etiketi (spell modu) ─────────────────────────────────
        if active and wrist_pt and not collect_mode:
            etxt = active
            (etw,eth),_ = cv2.getTextSize(etxt, FONT, 0.55, 1)
            ex = wrist_pt[0] - etw//2
            ey = wrist_pt[1] - 42
            rrect(frame, ex-8, ey-eth-6, ex+etw+8, ey+6, 6, (28,22,40))
            cv2.rectangle(frame, (ex-8,ey-eth-6), (ex+etw+8,ey+6), CYAN, 1)
            lbl(frame, etxt, ex, ey, 0.55, CYAN)

        # ── COLLECT modu içerik ───────────────────────────────────────────────
        if collect_mode:
            # Sol panel: accent + başlık
            cv2.line(frame, (PX+10,PY+1), (PX+PW-10,PY+1), GREEN, 2)
            lbl(frame, "GESTURE LIST", PX+14, PY+22, 0.52, GREY)

            for i,g in enumerate(ALL_GESTURES):
                gy  = PY + 38 + i*RH
                n   = counts[g]
                done = n >= TARGET
                act  = i == collect_idx
                nc  = GREEN if act else ((70,185,70) if done else GREY)
                dc  = GREEN if (act or done) else DIM
                cv2.circle(frame, (PX+16, gy+12), 5, dc, -1 if (act or done) else 1)
                lbl(frame, g, PX+28, gy+18, 0.52, nc, bold=act)
                fc = GREEN if act else ((55,145,55) if done else (42,75,52))
                pbar(frame, PX+14, gy+28, PW-28, 8, n/TARGET, fc)
                ctxt = f"{n}/{TARGET}"
                (ctw,_),_ = cv2.getTextSize(ctxt, FONT, 0.38, 1)
                lbl(frame, ctxt, PX+PW-ctw-10, gy+RH-8, 0.38, nc)

            # Kart fg
            gesture_name = ALL_GESTURES[collect_idx]
            n   = counts[gesture_name]
            done = n >= TARGET
            if   done:              acol = GREEN
            elif not hand_detected: acol = RED
            else:                   acol = CYAN
            cv2.rectangle(frame, (CX,CY), (CX+CW,CY+CH), acol, 2)
            cv2.line(frame, (CX+12,CY+1), (CX+CW-12,CY+1), acol, 2)

            if done:
                ll,lc = "TAMAMLANDI!", GREEN
            elif not hand_detected:
                ll,lc = "ELINI GOSTER", RED
            else:
                ll,lc = "BU GESTURE YAP, SONRA  [R]  YE BAS", GREY
            lbl(frame, ll, CX+18, CY+22, 0.46, lc)

            sc,tk = 1.85, 3
            gtext = gesture_name.upper()
            (gw,gh),_ = cv2.getTextSize(gtext, FONT, sc, tk)
            gx2 = CX+18;  gy2 = CY+22+gh+14
            tc  = GREEN if done else (RED if not hand_detected else CYAN)
            cv2.putText(frame, gtext, (gx2+2,gy2+2), FONT, sc, (0,0,0), tk+2)
            cv2.putText(frame, gtext, (gx2,gy2),      FONT, sc, tc, tk)
            pbar(frame, CX+18, CY+CH-42, CW-36, 14, n/TARGET,
                 GREEN if done else CYAN, r=5)
            hint = (f"{n}/{TARGET}   Tamamlandi!  Sonraki icin [D]" if done
                    else f"{n}/{TARGET}    [R]=Kaydet    [A/D]=Gesture sec    [C]=Spell moduna don")
            lbl(frame, hint, CX+18, CY+CH-10, 0.40, DIM)

            # Flash bildirim
            if time.time() < flash_until:
                (fw,_),_ = cv2.getTextSize(flash_msg, FONT, 1.0, 2)
                fx,fy = (w-fw)//2, h//2
                rrect(frame, fx-14, fy-36, fx+fw+14, fy+12, 8, (18,14,24))
                cv2.rectangle(frame, (fx-14,fy-36), (fx+fw+14,fy+12), CYAN, 1)
                lbl_c(frame, flash_msg, w//2, fy, 1.0, CYAN, bold=True)

            # Snapshot flaş
            if snap_alpha > 0.0:
                sov = frame.copy()
                cv2.rectangle(sov, (0,0), (w,h), (255,255,255), -1)
                cv2.addWeighted(sov, snap_alpha, frame, 1-snap_alpha, 0, frame)
                snap_alpha = max(0.0, snap_alpha - 0.10)

        # ── SPELL modu içerik ─────────────────────────────────────────────────
        else:
            # Sağ panel accent + başlık
            cv2.line(frame, (SPX+10,SPY+1), (SPX+SPW-10,SPY+1), CYAN, 2)
            lbl(frame, "SPELLS", SPX+14, SPY+22, 0.55, GREY)

            for i,spell in enumerate(SPELLS):
                seq  = spell["sequence"]
                prog = get_spell_progress(engine.buffer, seq)
                n_s  = len(seq)
                sy   = SPY + 38 + i*SRH
                act  = prog > 0
                nc   = WHITE if act else GREY
                lbl(frame, spell["name"], SPX+14, sy+16, 0.54, nc, bold=act)
                ptxt = f"{prog}/{n_s}"
                (ptw,_),_ = cv2.getTextSize(ptxt, FONT, 0.44, 1)
                lbl(frame, ptxt, SPX+SPW-ptw-12, sy+16, 0.44, CYAN if act else DIM)

                # Sequence chips
                cx2 = SPX+14;  cy2 = sy+26
                for j,g in enumerate(seq[:7]):
                    short   = ABBREV.get(g, g[:4])
                    matched = j < prog
                    bg2 = (35,70,35)   if matched else (30,26,40)
                    fg2 = (155,230,155) if matched else GREY
                    cw2,ch2 = chip(frame, cx2, cy2, short, bg2, fg2, scale=0.36)
                    cx2 += cw2 + 3
                    if j < len(seq)-1 and j < 6:
                        lbl(frame, ">", cx2, cy2+ch2-2, 0.28, DIM)
                        cx2 += 10

            # Buffer tokenlar
            buf   = list(engine.buffer[-8:])
            has_h = bool(engine.hold_gesture)
            tc2   = len(buf) + (1 if has_h else 0)
            if tc2 > 0 or engine.is_cooling_down:
                TW,TH,TG = 80, 34, 6
                tot2 = max(tc2,1)*(TW+TG)-TG
                bx2  = (w-tot2)//2;  by2 = h-TH-14

                for i,g in enumerate(buf):
                    x2 = bx2 + i*(TW+TG)
                    cooling = engine.is_cooling_down
                    bg3 = (26,18,18) if cooling else (22,46,22)
                    bc3 = (90,55,55) if cooling else (60,155,60)
                    tc3 = (175,115,115) if cooling else (170,230,170)
                    rrect(frame, x2, by2, x2+TW, by2+TH, 6, bg3)
                    cv2.rectangle(frame, (x2,by2), (x2+TW,by2+TH), bc3, 1)
                    abbr = ABBREV.get(g, g[:4])
                    lbl_c(frame, abbr, x2+TW//2, by2+TH-7, 0.44, tc3)

                if engine.hold_gesture and not engine.is_cooling_down:
                    x2   = bx2 + len(buf)*(TW+TG)
                    abbr = ABBREV.get(engine.hold_gesture, engine.hold_gesture[:4])
                    rrect(frame, x2, by2, x2+TW, by2+TH, 6, (26,26,40))
                    fill = int(TW * engine.hold_progress)
                    if fill > 8:
                        rrect(frame, x2, by2, x2+fill, by2+TH, 6, (28,75,170))
                    cv2.rectangle(frame, (x2,by2), (x2+TW,by2+TH), ORANGE, 1)
                    lbl_c(frame, abbr, x2+TW//2, by2+TH-7, 0.44, WHITE)

                if engine.is_cooling_down:
                    cd = f"COOLDOWN  {engine.cooldown_remaining:.1f}s"
                    lbl_c(frame, cd, w//2, by2-10, 0.62, RED, bold=True)

            # Spell aktivasyon efekti
            if active_spell:
                elapsed = time.time() - activation_time
                if elapsed < 2.0:
                    alpha_a = max(0.0, 1.0 - elapsed/2.0)
                    aov = frame.copy()
                    cv2.rectangle(aov, (0,0), (w,h), active_spell["color"], -1)
                    cv2.addWeighted(aov, alpha_a*0.38, frame, 1-alpha_a*0.38, 0, frame)
                    atxt = active_spell["name"].upper() + "!"
                    sc2,tk2 = 3.2, 5
                    (atw,ath),_ = cv2.getTextSize(atxt, FONT, sc2, tk2)
                    ax = (w-atw)//2;  ay = h//2+ath//2
                    cv2.putText(frame, atxt, (ax+3,ay+3), FONT, sc2, (0,0,0), tk2+2)
                    cv2.putText(frame, atxt, (ax,ay),     FONT, sc2, (255,255,255), tk2)

            # [C] ipucu alt orta
            htxt2 = "[C] Veri kayit modu"
            (htw2,_),_ = cv2.getTextSize(htxt2, FONT, 0.40, 1)
            lbl(frame, htxt2, (w-htw2)//2, h-4, 0.40, DIM)

        # FPS hesapla
        now       = time.time()
        fps_val   = 0.9*fps_val + 0.1*(1.0/(now-fps_time+1e-9))
        fps_time  = now

        # Ekrana tam doldur — aspect ratio korunarak crop-to-fill
        src_w, src_h = frame.shape[1], frame.shape[0]
        scale   = max(DISP_W / src_w, DISP_H / src_h)
        new_w   = int(src_w * scale)
        new_h   = int(src_h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        x0 = (new_w - DISP_W) // 2
        y0 = (new_h - DISP_H) // 2
        display = resized[y0:y0+DISP_H, x0:x0+DISP_W]
        cv2.imshow("Hand Capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:   # Q veya ESC
            break
        elif key == ord('c'):
            collect_mode = not collect_mode
            counts = load_counts()
        elif collect_mode:
            if key == ord('d'):
                collect_idx = (collect_idx+1) % len(ALL_GESTURES)
            elif key == ord('a'):
                collect_idx = (collect_idx-1) % len(ALL_GESTURES)
            elif key == ord('r'):
                if not hand_detected:
                    flash_msg   = "Once elini goster!"
                    flash_until = time.time() + 1.2
                elif cur_lm:
                    features = extract_features(cur_lm[0])
                    if features:
                        gesture_name = ALL_GESTURES[collect_idx]
                        write_header = not os.path.exists(CSV_PATH)
                        with open(CSV_PATH, "a", newline="") as f:
                            wr = csv.writer(f)
                            if write_header:
                                wr.writerow(["label"]+[f"f{i}" for i in range(42)])
                            wr.writerow([gesture_name]+features)
                        counts[gesture_name] += 1
                        flash_msg   = f"Kaydedildi!  {counts[gesture_name]}/{TARGET}"
                        flash_until = time.time() + 0.7
                        snap_alpha  = 1.0

cap.release()
cv2.destroyAllWindows()
