import time

SPELLS = [
    {"name": "Fireball",    "color": (0,   80,  255), "sequence": ["Fist", "Open Hand", "Point"]},
    {"name": "Shield",      "color": (200, 160,   0), "sequence": ["Peace", "Open Hand", "Peace"]},
    {"name": "Heal",        "color": (0,   200,  80), "sequence": ["OK", "Open Hand", "OK"]},
    {"name": "Darkness",    "color": (80,    0, 120), "sequence": ["Fist", "Rock", "Fist"]},
    {"name": "Blizzard",    "color": (200, 220, 255), "sequence": ["Open Hand", "Fist", "Open Hand", "Fist", "Peace"]},
    {"name": "Lightning",   "color": (255, 255,   0), "sequence": ["Rock", "Point", "Rock", "Point", "Rock"]},
    {"name": "Meteor",      "color": (0,   40,  255), "sequence": ["Thumbs Up", "Fist", "Open Hand", "Point", "Fist"]},
    {"name": "Apocalypse",  "color": (0,    0,  180), "sequence": ["Fist", "Open Hand", "Rock", "Peace", "Point",
                                                                     "Fist", "OK", "Thumbs Up", "Rock", "Fist"]},
]

MAX_BUFFER    = 10
HOLD_TIME     = 0.9
TIMEOUT       = 3.0
COOLDOWN_TIME = 2.0

ABBREV = {
    "Fist": "FIST", "Open Hand": "OPEN", "Thumbs Up": "THUP",
    "Point": "PONT", "Peace": "PEAC", "Rock": "ROCK",
    "Four": "FOUR", "OK": "OK",
}


def get_spell_progress(buffer, spell_seq):
    for k in range(len(spell_seq), 0, -1):
        if len(buffer) >= k and list(buffer[-k:]) == list(spell_seq[:k]):
            return k
    return 0


class SpellEngine:
    def __init__(self):
        self._buffer           = []
        self._hold_gesture     = None
        self._hold_start       = 0.0
        self._last_commit_time = time.time()
        self._cooldown_until   = 0.0

    def update(self, gesture):
        now = time.time()

        if now < self._cooldown_until:
            return None

        if now - self._last_commit_time > TIMEOUT and self._buffer:
            self._buffer.clear()

        if not gesture:
            self._hold_gesture = None
            return None

        if gesture != self._hold_gesture:
            self._hold_gesture = gesture
            self._hold_start   = now
            return None

        if now - self._hold_start < HOLD_TIME:
            return None

        # Commit
        self._buffer.append(gesture)
        self._last_commit_time = now
        self._hold_gesture     = None

        # Önce spell kontrolü — 10. gesture bir spell tamamlıyorsa FAILED değil spell dön
        spell = self._check_spells()
        if spell:
            return spell

        if len(self._buffer) >= MAX_BUFFER:
            self._buffer.clear()
            self._cooldown_until = now + COOLDOWN_TIME
            return "FAILED"

        return None

    def _check_spells(self):
        buf  = list(self._buffer)
        blen = len(buf)

        # Hangi spell'ler tamamen eşleşiyor?
        matches = [
            s for s in SPELLS
            if blen >= len(s["sequence"]) and buf[-len(s["sequence"]):] == list(s["sequence"])
        ]
        if not matches:
            return None

        # En uzun eşleşmeyi al
        best   = max(matches, key=lambda s: len(s["sequence"]))
        best_n = len(best["sequence"])

        # Daha uzun bir spell için buffer prefix oluşturuyor mu?
        # Eğer oluşturuyorsa henüz tetikleme, bekle
        for other in SPELLS:
            oseq = list(other["sequence"])
            on   = len(oseq)
            if on <= best_n:
                continue
            k = min(blen, on)
            if buf[-k:] == oseq[:k]:
                return None  # uzun spell hâlâ mümkün, bekle

        self._buffer.clear()
        return best

    @property
    def buffer(self):
        return list(self._buffer)

    @property
    def hold_progress(self):
        if not self._hold_gesture:
            return 0.0
        return min((time.time() - self._hold_start) / HOLD_TIME, 1.0)

    @property
    def hold_gesture(self):
        return self._hold_gesture

    @property
    def is_cooling_down(self):
        return time.time() < self._cooldown_until

    @property
    def cooldown_remaining(self):
        return max(0.0, self._cooldown_until - time.time())
