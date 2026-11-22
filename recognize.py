"""Real-time recognition. Press Q to quit."""
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp

from utils import normalize_landmarks

model = joblib.load("model.pkl")
hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
recent = deque(maxlen=10)   # majority vote over last 10 frames = less flicker

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    text = "No hand"
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
        feats = normalize_landmarks(hand).reshape(1, -1)
        recent.append(model.predict(feats)[0])
        conf = model.predict_proba(feats).max()
        text = f"{Counter(recent).most_common(1)[0][0]} ({conf:.0%})"
    else:
        recent.clear()
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
