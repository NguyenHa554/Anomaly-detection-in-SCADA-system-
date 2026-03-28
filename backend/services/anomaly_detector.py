"""
Runtime anomaly detector aligned with the current data2.ipynb logic.
"""

import json
import os
import pickle
import re
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

ML_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "ml_models")

# Selected features from the current notebook/export path.
SELECTED_FEATURES = [
    "FIT 101", "LIT 101", "MV 101", "P1_STATE", "P101 Status",
    "AIT 201", "AIT 202", "AIT 203", "FIT 201", "MV201",
    "P203 Status", "P205 Status",
    "AIT 301", "AIT 302", "AIT 303", "DPIT 301", "FIT 301",
    "LIT 301", "MV 301", "MV 302", "MV 303", "MV 304",
    "P3_STATE", "P301 Status",
    "AIT 402", "FIT 401", "LIT 401", "P401 Status", "UV401",
    "AIT 501", "AIT 502", "AIT 503", "AIT 504",
    "FIT 501", "FIT 502", "FIT 503", "FIT 504",
    "MV 501", "PIT 501", "PIT 502", "PIT 503",
    "FIT 601", "LSH 601", "P601 Status", "P102 Status",
    "LS 201", "LS 202", "LSL 203", "LSLL 203",
    "P2_STATE", "P201 Status", "P202 Status", "P204 Status",
    "P206 Status", "P207 Status", "P208 Status", "P302 Status",
    "AIT 401", "LS 401", "P4_STATE", "P402 Status",
    "P403 Status", "P404 Status", "MV 502", "MV 503",
    "MV 504", "P5_STATE", "P501 Status", "P502 Status",
    "LSH 602", "LSH 603", "LSL 601", "LSL 602",
    "LSL 603", "P6 STATE", "P602 Status", "P603 Status",
]


def get_stage_features(columns) -> Dict[str, list]:
    stages: Dict[str, list] = {}
    for stage_num in [2, 3, 4, 5, 6]:
        cols = [
            c for c in columns
            if re.search(rf"\b{stage_num}\d{{2}}\b", c)
            or re.fullmatch(rf"P{stage_num}_STATE", c)
        ]
        if cols:
            stages[f"P{stage_num}"] = cols
    return stages


STAGE_FEATURES: Dict[str, list] = get_stage_features(SELECTED_FEATURES)


def add_derivative_features(data: np.ndarray) -> np.ndarray:
    diff = np.diff(data, axis=0, prepend=data[0].reshape(1, -1))
    return np.concatenate([data, diff], axis=1)


class AnomalyDetector:
    def __init__(self, time_steps: int = 200):
        self.time_steps = time_steps
        self.models: Dict[str, object] = {}
        self.scalers: Dict[str, object] = {}
        self.stats: Dict[str, dict] = {}
        self.thresholds: Dict[str, dict] = {}
        self.buffers: Dict[str, list] = {stage: [] for stage in STAGE_FEATURES}
        self.violation_windows: Dict[str, Deque[bool]] = {}
        self._loaded = False
        self.loaded_stages = tuple(STAGE_FEATURES.keys())

    def load(self):
        """Load all saved models, scalers, thresholds, and z-score statistics."""
        import tensorflow as tf

        with open(os.path.join(ML_DIR, "stats.json")) as f:
            self.stats = json.load(f)
        with open(os.path.join(ML_DIR, "thresholds.json")) as f:
            self.thresholds = json.load(f)

        missing_thresholds = [stage for stage in STAGE_FEATURES if stage not in self.thresholds]
        if missing_thresholds:
            raise FileNotFoundError(
                f"Missing thresholds for stages: {', '.join(missing_thresholds)}"
            )

        self.violation_windows = {
            stage: deque(maxlen=int(self.thresholds[stage]["W"]))
            for stage in STAGE_FEATURES
        }
        self.buffers = {stage: [] for stage in STAGE_FEATURES}

        for stage in STAGE_FEATURES:
            model_path_keras = os.path.join(ML_DIR, f"{stage}_model.keras")
            model_path_h5 = os.path.join(ML_DIR, f"{stage}_model.h5")
            scaler_path = os.path.join(ML_DIR, f"{stage}_scaler.pkl")

            model_path = model_path_keras if os.path.exists(model_path_keras) else model_path_h5
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file not found for {stage}: {model_path_keras} or {model_path_h5}"
                )
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found for {stage}: {scaler_path}")

            self.models[stage] = tf.keras.models.load_model(model_path, compile=False)
            with open(scaler_path, "rb") as f:
                self.scalers[stage] = pickle.load(f)

        self._loaded = True
        self.loaded_stages = tuple(STAGE_FEATURES.keys())
        print("AnomalyDetector: all models loaded.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def add_data_point(self, stage: str, raw_features: np.ndarray):
        self.buffers[stage].append(raw_features)
        max_len = self.time_steps + 1
        if len(self.buffers[stage]) > max_len:
            self.buffers[stage].pop(0)

    def predict(self, stage: str) -> Optional[Dict]:
        buf = self.buffers[stage]
        if len(buf) < self.time_steps + 1:
            return {
                "stage": stage,
                "status": "warming_up",
                "buffer_fill": len(buf),
                "buffer_needed": self.time_steps + 1,
            }

        raw = np.array(buf[-(self.time_steps + 1):], dtype=np.float32)
        scaled = self.scalers[stage].transform(raw)
        enhanced = add_derivative_features(scaled)

        X = enhanced[:-1].reshape(1, self.time_steps, enhanced.shape[1])
        y_true = scaled[-1, : raw.shape[1]]
        y_pred = self.models[stage].predict(X, verbose=0)[0]

        error = np.abs(y_pred - y_true)
        mu = np.array(self.stats[stage]["mu"])
        sigma = np.array(self.stats[stage]["sigma"])
        z = (error - mu) / (sigma + 1e-8)
        max_z = float(np.max(z))

        threshold = float(self.thresholds[stage]["T"])
        window_size = int(self.thresholds[stage]["W"])

        window = self.violation_windows[stage]
        window.append(max_z > threshold)
        is_anomaly = (len(window) == window_size) and all(window)
        score = max_z / threshold

        return {
            "stage": stage,
            "status": "ok",
            "max_z_score": max_z,
            "threshold": threshold,
            "is_anomaly": is_anomaly,
            "anomaly_score": score,
        }


detector = AnomalyDetector()
