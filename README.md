# Real-Time Sign Language Recognition System

Recognizes a small set of static hand signs from a webcam in real time.

**Stack:** Python, OpenCV, MediaPipe, scikit-learn

## How it works
1. **MediaPipe Hands** detects 21 hand landmarks per frame.
2. Landmarks are normalized (wrist = origin, scaled) so position and hand size don't matter.
3. A **Random Forest** classifier predicts the sign from the 42 features.
4. Predictions are smoothed with a 10-frame majority vote and drawn on the live feed by **OpenCV**.

## Setup
```bash
pip install -r requirements.txt
python collect_data.py HELLO YES NO THANKS    # record samples per sign
python train.py                               # trains model, prints metrics
python recognize.py                           # live demo
```

## Results
- Signs: <list yours>
- Samples per sign: <fill in>
- Test accuracy: <paste from train.py output>
- Confusion matrix: see `confusion_matrix.png`

## Limitations
- Static, single-hand signs only (no motion-based or two-handed signs, no full sentences).
- Trained on one person's hand data, so accuracy may drop for other users and lighting.

## Future work
- Dynamic signs using landmark sequences (LSTM)
- More signers and signs
