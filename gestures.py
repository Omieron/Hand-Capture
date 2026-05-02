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

def get_finger_states(landmarks):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    fingers = []

    if landmarks[tips[0]].x < landmarks[pips[0]].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for i in range(1, 5):
        if landmarks[tips[i]].y < landmarks[pips[i]].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def detect_gesture(landmarks):
    f = get_finger_states(landmarks)

    if f == [0, 0, 0, 0, 0]:
        return "Fist"
    if f == [1, 1, 1, 1, 1]:
        return "Open Hand"
    if f == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    if f == [0, 1, 0, 0, 0]:
        return "Point"
    if f == [0, 1, 1, 0, 0]:
        return "Peace"
    if f == [0, 1, 0, 0, 1]:
        return "Rock"
    if f == [1, 1, 1, 1, 0]:
        return "Four"

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    if dist < 0.05:
        return "OK"

    return ""
