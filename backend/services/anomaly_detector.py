"""
anomaly_detector.py — Loads trained CNN models and runs inference.
Mirrors the prediction logic from data2.ipynb exactly.
"""

import json
import pickle
import numpy as np
import os
from typing import Dict, Optional

ML_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "ml_models")

# Stage → list of feature column names (order must match training)
STAGE_FEATURES: Dict[str, list] = {
    "P1": ["FIT 101", "LIT 101", "MV 101", "P1_STATE", "P101 Status",
           "AIT 201", "AIT 202", "AIT 203", "FIT 201", "MV201",
           "P203 Status", "P205 Status",
           "AIT 301", "AIT 302", "AIT 303", "DPIT 301", "FIT 301",
           "LIT 301", "MV 301", "MV 302", "MV 303", "MV 304",
           "P3_STATE", "P301 Status", "AIT 402"],
    "P2": ["AIT 201", "AIT 202", "AIT 203", "FIT 201", "MV201",
           "P203 Status", "P205 Status",
           "AIT 302", "AIT 303", "DPIT 301", "FIT 301", "LIT 301", "MV 301"],
    "P3": ["AIT 301", "AIT 302", "AIT 303", "DPIT 301", "FIT 301",
           "LIT 301", "MV 301", "MV 302", "MV 303", "MV 304",
           "P3_STATE", "P301 Status", "AIT 402", "FIT 401", "LIT 401",
           "P401 Status", "UV401"],
    "P5": ["AIT 501", "AIT 502", "AIT 503", "AIT 504",
           "FIT 501", "FIT 502", "FIT 503", "FIT 504",
           "MV 501", "PIT 501", "PIT 502", "PIT 503", "FIT 601"],
}


def add_derivative_features(data: np.ndarray) -> np.ndarray:
    diff = np.diff(data, axis=0, prepend=data[0].reshape(1, -1))
    return np.concatenate([data, diff], axis=1)


"""
anomaly_detector.py — Loads trained CNN models and runs inference.
Mirrors the prediction logic from data2.ipynb exactly.
"""

import json
import pickle
import numpy as np
import os
from collections import deque
from typing import Deque, Dict, Optional

ML_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "ml_models")

# Stage → list of feature column names (order must match training)
STAGE_FEATURES: Dict[str, list] = {
    "P1": ["FIT 101", "LIT 101", "MV 101", "P1_STATE", "P101 Status",
           "AIT 201", "AIT 202", "AIT 203", "FIT 201", "MV201",
           "P203 Status", "P205 Status",
           "AIT 301", "AIT 302", "AIT 303", "DPIT 301", "FIT 301",
           "LIT 301", "MV 301", "MV 302", "MV 303", "MV 304",
           "P3_STATE", "P301 Status", "AIT 402"],
    "P2": ["AIT 201", "AIT 202", "AIT 203", "FIT 201", "MV201",
           "P203 Status", "P205 Status",
           "AIT 302", "AIT 303", "DPIT 301", "FIT 301", "LIT 301", "MV 301"],
    "P3": ["AIT 301", "AIT 302", "AIT 303", "DPIT 301", "FIT 301",
           "LIT 301", "MV 301", "MV 302", "MV 303", "MV 304",
           "P3_STATE", "P301 Status", "AIT 402", "FIT 401", "LIT 401",
           "P401 Status", "UV401"],
    "P5": ["AIT 501", "AIT 502", "AIT 503", "AIT 504",
           "FIT 501", "FIT 502", "FIT 503", "FIT 504",
           "MV 501", "PIT 501", "PIT 502", "PIT 503", "FIT 601"],
}


def add_derivative_features(data: np.ndarray) -> np.ndarray:
    diff = np.diff(data, axis=0, prepend=data[0].reshape(1, -1))
    return np.concatenate([data, diff], axis=1)


def create_sequence(enhanced: np.ndarray, time_steps: int = 200):
    """Return the last `time_steps` rows as a (1, time_steps, features) tensor."""
    seq = enhanced[-time_steps:]
    return seq.reshape(1, time_steps, enhanced.shape[1])


class AnomalyDetector:
    def __init__(self, time_steps: int = 200):
        self.time_steps = time_steps
        self.models: Dict = {}
        self.scalers: Dict = {}
        self.stats: Dict = {}
        self.thresholds: Dict = {}
        self.buffers: Dict[str, list] = {s: [] for s in STAGE_FEATURES}
        # Rolling windows of threshold violations to mirror notebook logic:
        # anomaly = 1 only if last W max_z values are all > T.
        self.violation_windows: Dict[str, Deque[bool]] = {}
        self._loaded = False

    def load(self):
        """Load all models, scalers, and statistics from disk."""
        import tensorflow as tf

        # stats & thresholds
        with open(os.path.join(ML_DIR, "stats.json")) as f:
            self.stats = json.load(f)
        with open(os.path.join(ML_DIR, "thresholds.json")) as f:
            self.thresholds = json.load(f)
        # Initialize rolling windows after thresholds are available
        self.violation_windows = {
            stage: deque(maxlen=int(self.thresholds[stage]["W"]))
            for stage in STAGE_FEATURES
        }

        for stage in STAGE_FEATURES:
            model_path_keras = os.path.join(ML_DIR, f"{stage}_model.keras")
            model_path_h5 = os.path.join(ML_DIR, f"{stage}_model.h5")
            scaler_path = os.path.join(ML_DIR, f"{stage}_scaler.pkl")

            # Prefer modern Keras format; fallback to legacy .h5 models.
            model_path = model_path_keras if os.path.exists(model_path_keras) else model_path_h5
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found for {stage}: {model_path_keras} or {model_path_h5}")

            # Inference only: skip training config deserialization for compatibility.
            self.models[stage] = tf.keras.models.load_model(model_path, compile=False)
            with open(scaler_path, "rb") as f:
                self.scalers[stage] = pickle.load(f)

        self._loaded = True
        print("AnomalyDetector: all models loaded.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def add_data_point(self, stage: str, raw_features: np.ndarray):
        """Append a single raw feature vector to the rolling buffer."""
        self.buffers[stage].append(raw_features)
        # Keep only what's needed
        max_len = self.time_steps + 1
        if len(self.buffers[stage]) > max_len:
            self.buffers[stage].pop(0)

    def predict(self, stage: str) -> Optional[Dict]:
        """
        Run prediction for one stage.
        Returns None if buffer not warm yet.
        """
        buf = self.buffers[stage]
        if len(buf) < self.time_steps + 1:
            return {
                "stage": stage,
                "status": "warming_up",
                "buffer_fill": len(buf),
                "buffer_needed": self.time_steps + 1,
            }

        # Scale
        raw = np.array(buf[-(self.time_steps + 1):])
        scaled = self.scalers[stage].transform(raw)

        # Derivative features
        enhanced = add_derivative_features(scaled)

        # Sequence: first time_steps rows → predict (time_steps+1)-th
        X = enhanced[:-1].reshape(1, self.time_steps, enhanced.shape[1])
        y_true = scaled[-1, : raw.shape[1]]

        y_pred = self.models[stage].predict(X, verbose=0)[0]

        # Reconstruction error & z-score
        error  = np.abs(y_pred - y_true)
        mu     = np.array(self.stats[stage]["mu"])
        sigma  = np.array(self.stats[stage]["sigma"])
        z      = (error - mu) / (sigma + 1e-8)
        max_z  = float(np.max(z))

        T = self.thresholds[stage]["T"]
        W = self.thresholds[stage]["W"]

        # Only flag if last W max_z values are all > T (notebook rolling window logic)
        window = self.violation_windows[stage]
        window.append(max_z > T)
        is_anomaly = (len(window) == W) and all(window)
        score      = max_z / T  # normalised; >1 means T exceeded

        return {
            "stage":         stage,
            "status":        "ok",
            "max_z_score":   max_z,
            "threshold":     T,
            "is_anomaly":    is_anomaly,
            "anomaly_score": score,
        }


# Singleton used throughout the backend
detector = AnomalyDetector()
