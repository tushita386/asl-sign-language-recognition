
"""Train and evaluate the 24-class static ASL handshape classifier."""

import os
import shutil

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

DATA_PATH = "data/landmarks.csv"
MODEL_PATH = "model.pkl"

LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
FEATURES = [f"f{i}" for i in range(42)]

# Load dataset
df = pd.read_csv(DATA_PATH)

required = ["label"] + FEATURES
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

if df[required].isna().any().any():
    raise ValueError("Dataset contains missing feature or label values.")

# Keep metadata out of model training
X = df[FEATURES]
y = df["label"]

# Validate classes
found = set(y.unique())
expected = set(LABELS)

if found != expected:
    raise ValueError(
        f"Class mismatch. Missing: {sorted(expected - found)}, "
        f"Unexpected: {sorted(found - expected)}"
    )

print("Dataset shape:", X.shape)
print("\nSamples per class:")
print(y.value_counts().sort_index())

# Random frame-level split: baseline evaluation only
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

# Train classifier
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=LABELS,
        zero_division=0,
    )
)

print(f"Random-split test accuracy: {accuracy:.4f}")

# Preserve previous model if it exists
if os.path.exists(MODEL_PATH):
    shutil.copy2(MODEL_PATH, "model_previous.pkl")
    print("Previous model backed up as model_previous.pkl")

# Save trained model
joblib.dump(model, MODEL_PATH)

# Generate confusion matrix
fig, ax = plt.subplots(figsize=(14, 12))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=LABELS,
    xticks_rotation=45,
    cmap="Blues",
    ax=ax,
    colorbar=False,
)

ax.set_title("ASL Static Handshape Recognition - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

print("\nSaved model.pkl")
print("Saved confusion_matrix.png")
print("\nTraining completed successfully!")
