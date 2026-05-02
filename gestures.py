from collections import deque

ALL_GESTURES = [
    "Fist",
    "Open Hand",
    "Thumbs Up",
    "Point",
    "Peace",
    "Rock",
    "Four",
    "OK",
]

_buffer = deque(maxlen=12)


def get_finger_states(landmarks):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    # Başparmak: elin yönüne göre
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    hand_facing_right = index_mcp.x > wrist.x
    if hand_facing_right:
        thumb = 1 if landmarks[4].x > landmarks[3].x else 0
    else:
        thumb = 1 if landmarks[4].x < landmarks[3].x else 0

    # Diğer 4 parmak: tip, pip'in üzerinde mi
    others = [1 if landmarks[tips[i]].y < landmarks[pips[i]].y else 0
              for i in range(1, 5)]

    return [thumb] + others


def detect_gesture(landmarks):
    f = get_finger_states(landmarks)

    # OK: başparmak ucu ile işaret ucu mesafesi
    t, i = landmarks[4], landmarks[8]
    dist = ((t.x - i.x) ** 2 + (t.y - i.y) ** 2) ** 0.5
    if dist < 0.05 and f[1] == 0:
        return "OK"

    if f == [0, 0, 0, 0, 0]: return "Fist"
    if f == [1, 1, 1, 1, 1]: return "Open Hand"
    if f == [1, 0, 0, 0, 0]: return "Thumbs Up"
    if f == [0, 1, 0, 0, 0]: return "Point"
    if f == [0, 1, 1, 0, 0]: return "Peace"
    if f == [0, 1, 0, 0, 1]: return "Rock"
    if f == [1, 1, 1, 1, 0]: return "Four"

    return ""


def smooth_gesture(landmarks):
    raw = detect_gesture(landmarks)
    _buffer.append(raw)

    if len(_buffer) < 6:
        return raw

    counts = {}
    for g in _buffer:
        counts[g] = counts.get(g, 0) + 1

    best = max(counts, key=counts.get)
    if counts[best] >= len(_buffer) * 0.6 and best != "":
        return best

    return ""
