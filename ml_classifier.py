"""
ml_classifier.py
-----------------
Loads the trained model + label encoder (from train_model.py) and
exposes a predict() method that takes normalized landmarks and returns
a (label, confidence, is_improper) result -- mirroring the shape of the
rule-based path so main.py can switch between the two easily.
"""

import os
from typing import List, Optional, Tuple

import joblib
import numpy as np

import config


class MLGestureClassifier:
    def __init__(self):
        if not os.path.isfile(config.MODEL_PATH) or not os.path.isfile(config.LABEL_ENCODER_PATH):
            raise FileNotFoundError(
                f"Model files not found at '{config.MODEL_PATH}'. "
                "Run collect_data.py then train_model.py first, or set "
                "USE_ML_MODEL = False in config.py."
            )
        self.model = joblib.load(config.MODEL_PATH)
        self.encoder = joblib.load(config.LABEL_ENCODER_PATH)

    def predict(self, landmarks_norm: List[Tuple[float, float]]) -> Optional[dict]:
        """
        landmarks_norm: list of 21 (x, y) normalized points.
        Returns a dict shaped like a config.GESTURE_RULES entry so
        main.py can treat ML and rule-based results the same way:
            {"name": ..., "label": ..., "improper": bool, "confidence": float}
        or None if confidence is below threshold.
        """
        features = np.array([coord for point in landmarks_norm for coord in point]).reshape(1, -1)

        probs = self.model.predict_proba(features)[0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        label_name = self.encoder.inverse_transform([best_idx])[0]

        if confidence < config.ML_CONFIDENCE_THRESHOLD:
            return {
                "name": "uncertain",
                "label": f"Uncertain ({confidence:.0%})",
                "improper": False,
                "confidence": confidence,
            }

        return {
            "name": label_name,
            "label": f"{label_name} ({confidence:.0%})",
            "improper": label_name in config.ML_IMPROPER_LABELS,
            "confidence": confidence,
        }