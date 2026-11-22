"""Real-time sign recognition with a side-panel UI and sign-to-text builder.

Hold a sign steady for ~1.5 s to add it to the sentence.
Keys:  SPACE = add a space   BACKSPACE = delete last char
       C = clear sentence    Q = quit
"""
import time
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np

from utils import normalize_landmarks

# ---------------- settings ----------------
MODEL_PATH = "model.pkl"
CONF_THRESHOLD = 0.70    # below this the UI shows "Unsure"
HOLD_SECONDS = 1.5       # how long to hold a sign to type it
SMOOTHING = 10           # frames averaged to reduce flicker
PANEL_W = 360
DISPLAY = {"ILOVEYOU": "I LOVE YOU"}   # nicer names for multi-word labels

# colors (BGR)
BG = (38, 32, 30)
CARD = (56, 48, 44)
TEXT = (240, 240, 240)
MUTED = (160, 160, 160)
GREEN = (110, 200, 90)
AMBER = (60, 180, 240)
RED = (80, 80, 230)
ACCENT = (230, 160, 70)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put(img, text, pos, scale=0.6, color=TEXT, thick=1):
    cv2.putText(img, text, pos, FONT, scale, color, thick, cv2.LINE_AA)


def bar(img, x, y, w, h, frac, color):
    frac = float(np.clip(frac, 0, 1))
    cv2.rectangle(img, (x, y), (x + w, y + h), CARD, -1)
    cv2.rectangle(img, (x, y), (x + int(w * frac), y + h), color, -1)


def hand_box(hand, w, h, pad=20):
    xs = [lm.x * w for lm in hand.landmark]
    ys = [lm.y * h for lm in hand.landmark]
    return (max(int(min(xs)) - pad, 0), max(int(min(ys)) - pad, 0),
            min(int(max(xs)) + pad, w - 1), min(int(max(ys)) + pad, h - 1))


def wrap(text, width=20):
    return [text[i:i + width] for i in range(0, len(text), width)] or [""]


def build_panel(h, classes, avg, label, conf, hold_frac, sentence, fps):
    panel = np.full((h, PANEL_W, 3), BG, dtype=np.uint8)
    pad = 20
    inner = PANEL_W - 2 * pad

    put(panel, "SIGN RECOGNITION", (pad, 34), 0.7, ACCENT, 2)
    put(panel, f"{fps:4.0f} FPS", (PANEL_W - 95, 34), 0.5, MUTED)

    # big prediction
    cv2.rectangle(panel, (pad, 52), (PANEL_W - pad, 150), CARD, -1)
    if label is not None:
        shown = DISPLAY.get(label, label)
        put(panel, shown, (pad + 14, 112), 1.5 if len(shown) < 4 else 0.9, GREEN, 3)
    elif avg is not None:
        put(panel, "Unsure", (pad + 14, 112), 1.1, AMBER, 2)
    else:
        put(panel, "No hand", (pad + 14, 112), 1.0, MUTED, 2)

    # confidence
    conf_val = conf if avg is not None else 0.0
    put(panel, f"Confidence  {conf_val:.0%}", (pad, 180), 0.5, MUTED)
    bar(panel, pad, 190, inner, 12, conf_val, GREEN if conf_val >= CONF_THRESHOLD else RED)

    # hold-to-type progress
    put(panel, "Hold to type", (pad, 228), 0.5, MUTED)
    bar(panel, pad, 238, inner, 12, hold_frac, ACCENT)

    # Top 5 predictions only, so all 24 classes fit the panel
    put(panel, "Top predictions", (pad, 280), 0.5, MUTED)

    y = 296
    if avg is not None:
        top_indices = np.argsort(avg)[::-1][:5]

        for i in top_indices:
            name = classes[i]
            p = float(avg[i])

            put(panel, name, (pad, y + 11), 0.45, TEXT)
            bar(panel, pad + 95, y, inner - 95, 12, p, ACCENT)
            y += 24

    # sentence
    sy = 430
    put(panel, "Text", (pad, sy), 0.5, MUTED)
    cv2.rectangle(panel, (pad, sy + 8), (PANEL_W - pad, h - 40), CARD, -1)
    for j, line in enumerate(wrap(sentence)[-4:]):
        put(panel, line, (pad + 10, sy + 36 + j * 28), 0.7, TEXT, 2)

    put(panel, "SPACE  BKSP  C=clear  Q=quit", (pad, h - 14), 0.42, MUTED)
    return panel


def main():
    model = joblib.load(MODEL_PATH)
    classes = list(model.classes_)

    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    history = deque(maxlen=SMOOTHING)
    sentence = ""
    hold_label, hold_start, committed = None, 0.0, False
    prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        avg, label, conf = None, None, 0.0
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            feats = normalize_landmarks(hand).reshape(1, -1)
            history.append(model.predict_proba(feats)[0])
            avg = np.mean(history, axis=0)
            i = int(avg.argmax())
            conf = float(avg[i])
            if conf >= CONF_THRESHOLD:
                label = classes[i]

            draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            x1, y1, x2, y2 = hand_box(hand, w, h)
            color = GREEN if label else AMBER
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = DISPLAY.get(label, label) if label else "Unsure"
            cv2.rectangle(frame, (x1, max(y1 - 28, 0)), (x1 + 12 * len(tag) + 16, y1), color, -1)
            put(frame, tag, (x1 + 8, max(y1 - 8, 16)), 0.6, (20, 20, 20), 2)
        else:
            history.clear()

        # hold-to-type logic
        now = time.time()
        if label is not None and label == hold_label:
            hold_frac = (now - hold_start) / HOLD_SECONDS
            if hold_frac >= 1 and not committed:
                token = label if len(label) == 1 else f" {DISPLAY.get(label, label)} "
                sentence += token
                committed = True
            hold_frac = 1.0 if committed else hold_frac
        else:
            hold_label, hold_start, committed, hold_frac = label, now, False, 0.0

        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now

        panel = build_panel(h, classes, avg, label, conf, hold_frac, sentence, fps)
        cv2.imshow("Sign Language Recognition", np.hstack([frame, panel]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            sentence += " "
        elif key in (8, 127):          # backspace / delete
            sentence = sentence[:-1]
        elif key == ord("c"):
            sentence = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
