import numpy as np


def normalize_landmarks(hand_landmarks):
    """Turn 21 MediaPipe landmarks into a 42-value feature vector that does
    not depend on where the hand is in the frame or how big it appears."""
    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
    pts = pts - pts[0]                      # wrist becomes the origin
    scale = np.max(np.abs(pts)) or 1.0      # avoid divide-by-zero
    return (pts / scale).flatten()
