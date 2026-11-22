
"""Collect static ASL alphabet hand landmark samples."""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp

from utils import normalize_landmarks

LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

FIELDS = (
    ["person_id", "session_id", "timestamp", "label"]
    + [f"f{i}" for i in range(42)]
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", required=True, help="Your signer ID")
    parser.add_argument("--session", required=True, help="Recording session ID")
    parser.add_argument("--output", default="data/landmarks.csv")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--interval", type=float, default=0.05)
    args = parser.parse_args()

    if args.samples < 1 or args.interval <= 0:
        parser.error("Samples and interval must be positive.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    new_file = not os.path.exists(args.output) or os.path.getsize(args.output) == 0

    hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions.")

    idx = 0
    recording = False
    counts = {label: 0 for label in LABELS}
    last_sample = 0.0

    try:
        with open(args.output, "a", newline="") as file:
            writer = csv.writer(file)

            if new_file:
                writer.writerow(FIELDS)

            while True:
                ok, frame = cap.read()

                if not ok:
                    print("Could not read camera frame.")
                    break

                frame = cv2.flip(frame, 1)
                result = hands.process(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )

                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    draw.draw_landmarks(
                        frame,
                        hand,
                        mp.solutions.hands.HAND_CONNECTIONS,
                    )

                    now = time.monotonic()

                    if recording and now - last_sample >= args.interval:
                        features = normalize_landmarks(hand)

                        if len(features) == 42:
                            writer.writerow([
                                args.person,
                                args.session,
                                datetime.now().isoformat(timespec="milliseconds"),
                                LABELS[idx],
                                *features,
                            ])

                            counts[LABELS[idx]] += 1
                            last_sample = now

                status = (
                    f"{LABELS[idx]} | "
                    f"{'REC' if recording else 'PAUSED'} | "
                    f"n={counts[LABELS[idx]]}/{args.samples}"
                )

                cv2.putText(
                    frame,
                    status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("ASL Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                elif key == ord(" "):
                    if not recording:
                        if counts[LABELS[idx]] >= args.samples:
                            print("Target already reached.")
                        else:
                            recording = True
                            last_sample = 0.0
                    else:
                        recording = False
                        print(
                            f"{LABELS[idx]}: "
                            f"{counts[LABELS[idx]]} samples"
                        )

                elif key == ord("n"):
                    recording = False

                    if idx < len(LABELS) - 1:
                        idx += 1
                    else:
                        print("All static alphabet signs completed.")
                        break

    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

    print("\nCollection summary:")
    for label, count in counts.items():
        print(f"{label}: {count} samples")


if __name__ == "__main__":
    main()
