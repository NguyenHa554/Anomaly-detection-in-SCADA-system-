"""
export_models.py - Retrains stage models and calibrates thresholds.

Flow:
1. Train models on the clean pre-attack segment only.
2. Hold out the tail of that clean segment as validation-normal data.
3. Choose thresholds from clean validation behavior.
4. Report attack-detection metrics on the attacked test split.

Run once before starting the backend:
    python export_models.py
"""

import json
import os
import pickle
import re
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential

# Config
DATA_FILE = "SWaT.csv"
OUTPUT_DIR = "ml_models"
TIME_STEPS = 200
TRAIN_TEST_SPLIT_TS = "2019-07-20 07:00:00"
CLEAN_VALIDATION_FRACTION = 0.2
VALIDATION_EXTENSION_SECONDS = 600
MAX_VALIDATION_RAW_EXCEEDANCE_RATE = 0.01
SEED = 42

BASE_THRESHOLD_GRID = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0)
WINDOW_GRID = (10, 20, 30, 45, 60, 90, 120, 150, 200, 300)
VALIDATION_THRESHOLD_QUANTILES = (95.0, 97.5, 99.0, 99.5, 99.8, 99.9, 99.95)

# Features selected by correlation with "Attack" in the notebook
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

ATTACK_PERIODS_UTC = [
    ("2019-07-20 07:08:46", "2019-07-20 07:10:31"),
    ("2019-07-20 07:15:00", "2019-07-20 07:19:32"),
    ("2019-07-20 07:26:57", "2019-07-20 07:30:48"),
    ("2019-07-20 07:38:50", "2019-07-20 07:46:20"),
    ("2019-07-20 07:54:00", "2019-07-20 07:56:00"),
    ("2019-07-20 08:02:56", "2019-07-20 08:16:18"),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


def add_derivative_features(data: np.ndarray) -> np.ndarray:
    diff = np.diff(data, axis=0, prepend=data[0].reshape(1, -1))
    return np.concatenate([data, diff], axis=1)


def create_sequences(X: np.ndarray, time_steps: int = TIME_STEPS) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    original_features_count = X.shape[1] // 2
    for i in range(len(X) - time_steps):
        Xs.append(X[i : i + time_steps])
        ys.append(X[i + time_steps, :original_features_count])
    return np.array(Xs), np.array(ys)


def get_stage_features(columns: Iterable[str]) -> Dict[str, list]:
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


def build_model(time_steps: int, n_in: int, n_out: int) -> tf.keras.Model:
    return Sequential(
        [
            Input(shape=(time_steps, n_in)),
            Dense(n_in * 3),
            Conv1D(32, 2, activation="relu", padding="same"),
            BatchNormalization(),
            Conv1D(32, 2, activation="relu", padding="same"),
            MaxPooling1D(2),
            Conv1D(64, 2, activation="relu", padding="same"),
            BatchNormalization(),
            Conv1D(64, 2, activation="relu", padding="same"),
            MaxPooling1D(2),
            Conv1D(128, 2, activation="relu", padding="same"),
            BatchNormalization(),
            Conv1D(128, 2, activation="relu", padding="same"),
            MaxPooling1D(2),
            Conv1D(256, 2, activation="relu", padding="same"),
            Flatten(),
            Dropout(0.3),
            Dense(n_out),
        ]
    )


def get_attack_intervals(y_true: np.ndarray) -> list[Tuple[int, int]]:
    diff = np.diff(y_true, prepend=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    if len(starts) > len(ends):
        ends = np.append(ends, len(y_true) - 1)
    return list(zip(starts, ends))


def extend_attack_labels(y_true: np.ndarray, extension_seconds: int = VALIDATION_EXTENSION_SECONDS) -> np.ndarray:
    y_extended = y_true.copy()
    for _, end in get_attack_intervals(y_true):
        extended_end = min(end + extension_seconds, len(y_true) - 1)
        y_extended[end + 1 : extended_end + 1] = 1
    return y_extended


def compute_attack_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def detect_episodes(max_z: np.ndarray, threshold: float, window_size: int) -> np.ndarray:
    s = pd.Series(max_z)
    return (
        s.rolling(window_size)
        .apply(lambda x: 1 if (x > threshold).all() else 0, raw=True)
        .fillna(0)
        .astype(int)
        .to_numpy()
    )


def count_episode_starts(pred: np.ndarray) -> int:
    if pred.size == 0:
        return 0
    prev = np.concatenate(([0], pred[:-1]))
    return int(np.sum((pred == 1) & (prev == 0)))


def split_clean_train_validation(df_train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df_train) * (1.0 - CLEAN_VALIDATION_FRACTION))
    min_rows = TIME_STEPS + 50
    split_idx = max(split_idx, min_rows)
    split_idx = min(split_idx, len(df_train) - min_rows)
    fit_df = df_train.iloc[:split_idx].copy()
    val_df = df_train.iloc[split_idx:].copy()
    if len(fit_df) <= TIME_STEPS or len(val_df) <= TIME_STEPS:
        raise ValueError("Not enough clean rows to create train/validation splits.")
    return fit_df, val_df


def threshold_candidates(max_z: np.ndarray) -> list[float]:
    values = set(BASE_THRESHOLD_GRID)
    for q in VALIDATION_THRESHOLD_QUANTILES:
        values.add(round(float(np.percentile(max_z, q)), 3))
    return sorted(v for v in values if v > 0)


def calibrate_threshold(stage: str, validation_max_z: np.ndarray) -> Dict[str, float]:
    best_safe = None
    best_fallback = None

    for threshold in threshold_candidates(validation_max_z):
        raw_exceedance_rate = float(np.mean(validation_max_z > threshold))
        if raw_exceedance_rate > MAX_VALIDATION_RAW_EXCEEDANCE_RATE:
            continue

        for window_size in WINDOW_GRID:
            pred = detect_episodes(validation_max_z, threshold, window_size)
            alert_points = int(pred.sum())
            episode_starts = count_episode_starts(pred)
            candidate = {
                "T": float(threshold),
                "W": int(window_size),
                "validation_raw_exceedance_rate": raw_exceedance_rate,
                "validation_alert_points": alert_points,
                "validation_episode_starts": episode_starts,
            }

            if best_fallback is None or (
                candidate["validation_alert_points"],
                candidate["validation_raw_exceedance_rate"],
                candidate["T"],
                candidate["W"],
            ) < (
                best_fallback["validation_alert_points"],
                best_fallback["validation_raw_exceedance_rate"],
                best_fallback["T"],
                best_fallback["W"],
            ):
                best_fallback = candidate

            if alert_points == 0:
                if best_safe is None or (candidate["T"], candidate["W"]) < (best_safe["T"], best_safe["W"]):
                    best_safe = candidate
                break

    selected = best_safe if best_safe is not None else best_fallback
    if selected is None:
        raise ValueError(f"Failed to calibrate threshold for {stage}.")

    selected["selection"] = "clean_validation_safe" if best_safe is not None else "fallback_min_alerts"
    return selected


def score_max_z(model, enhanced_data: np.ndarray, true_scaled: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    X_seq, y_seq = create_sequences(enhanced_data, TIME_STEPS)
    pred = model.predict(X_seq, verbose=0)
    err = np.abs(pred - y_seq)
    z = (err - mu) / (sigma + 1e-8)
    return np.max(z, axis=1)


def prepare_dataframe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading data ...")
    df = pd.read_csv(DATA_FILE, header=1, low_memory=False)

    time_cands = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
    ts_col = time_cands[0] if time_cands else df.columns[0]
    parsed_ts = pd.to_datetime(df[ts_col], format="ISO8601", errors="coerce", utc=True)
    df = df.loc[parsed_ts.notna()].copy()
    df[ts_col] = parsed_ts.loc[parsed_ts.notna()]
    df = df.set_index(ts_col)

    for col in df.columns:
        if df[col].astype(str).str.contains("Active|Inactive", case=False, na=False).any():
            df[col] = df[col].map({"Active": 1, "Inactive": 0})

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Attack"] = 0
    for start_text, end_text in ATTACK_PERIODS_UTC:
        start = pd.to_datetime(start_text).tz_localize("UTC")
        end = pd.to_datetime(end_text).tz_localize("UTC")
        df.loc[start:end, "Attack"] = 1

    df_model = df[SELECTED_FEATURES + ["Attack"]].copy()
    df_model.ffill(inplace=True)
    df_model.bfill(inplace=True)

    split_ts = pd.to_datetime(TRAIN_TEST_SPLIT_TS).tz_localize("UTC")
    df_train = df_model.loc[df_model.index <= split_ts]
    df_test = df_model.loc[df_model.index > split_ts]

    print(f"Train rows: {len(df_train)} (attacks={int(df_train['Attack'].sum())})")
    print(f"Test rows : {len(df_test)} (attacks={int(df_test['Attack'].sum())})")
    if int(df_train["Attack"].sum()) != 0:
        raise ValueError("Training split must be clean normal data.")
    if int(df_test["Attack"].sum()) == 0:
        raise ValueError("Test split must include attacks for evaluation.")

    return df_train, df_test


def main():
    df_train_full, df_test = prepare_dataframe()
    df_fit, df_val = split_clean_train_validation(df_train_full)
    print(f"Clean fit rows: {len(df_fit)}")
    print(f"Clean val rows: {len(df_val)}")

    X_fit = df_fit[SELECTED_FEATURES]
    X_val = df_val[SELECTED_FEATURES]
    X_test = df_test[SELECTED_FEATURES]
    y_test = df_test["Attack"].to_numpy(dtype=int)

    stage_mapping = get_stage_features(X_fit.columns)
    print(f"Stages found: {list(stage_mapping.keys())}")

    stats_output = {}
    threshold_output = {}

    print("\nStage  | T      W    | ValExc  TestPrec TestRec TestF1")

    for stage, cols in stage_mapping.items():
        print(f"\n{'=' * 55}\n  Stage: {stage} ({len(cols)} features)\n{'=' * 55}")

        scaler = MinMaxScaler()
        fit_scaled = scaler.fit_transform(X_fit[cols].values)
        val_scaled = scaler.transform(X_val[cols].values)
        test_scaled = scaler.transform(X_test[cols].values)

        fit_enhanced = add_derivative_features(fit_scaled)
        val_enhanced = add_derivative_features(val_scaled)
        test_enhanced = add_derivative_features(test_scaled)

        X_seq_fit, y_seq_fit = create_sequences(fit_enhanced, TIME_STEPS)
        if X_seq_fit.size == 0:
            raise ValueError(f"Not enough fit rows to create sequences for {stage}.")

        model = build_model(TIME_STEPS, X_seq_fit.shape[2], y_seq_fit.shape[1])
        model.compile(optimizer="adam", loss="mse")

        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model.fit(
            X_seq_fit,
            y_seq_fit,
            epochs=100,
            batch_size=64,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=1,
        )

        fit_pred = model.predict(X_seq_fit, verbose=0)
        fit_err = np.abs(fit_pred - y_seq_fit)
        mu = np.mean(fit_err, axis=0)
        sigma = np.std(fit_err, axis=0)
        stats_output[stage] = {"mu": mu.tolist(), "sigma": sigma.tolist()}

        validation_max_z = score_max_z(model, val_enhanced, val_scaled, mu, sigma)
        threshold_cfg = calibrate_threshold(stage, validation_max_z)

        test_max_z = score_max_z(model, test_enhanced, test_scaled, mu, sigma)
        y_test_aligned = y_test[TIME_STEPS:]
        y_test_extended = extend_attack_labels(y_test_aligned, VALIDATION_EXTENSION_SECONDS)
        test_pred = detect_episodes(test_max_z, threshold_cfg["T"], threshold_cfg["W"])
        attack_metrics = compute_attack_metrics(y_test_extended, test_pred)

        threshold_cfg.update(
            {
                "attack_precision": attack_metrics["precision"],
                "attack_recall": attack_metrics["recall"],
                "attack_f1": attack_metrics["f1"],
                "test_max_z_mean": float(np.mean(test_max_z)),
                "test_max_z_p95": float(np.percentile(test_max_z, 95)),
                "validation_max_z_mean": float(np.mean(validation_max_z)),
                "validation_max_z_p95": float(np.percentile(validation_max_z, 95)),
            }
        )
        threshold_output[stage] = threshold_cfg

        model_path = os.path.join(OUTPUT_DIR, f"{stage}_model.keras")
        scaler_path = os.path.join(OUTPUT_DIR, f"{stage}_scaler.pkl")
        model.save(model_path)
        with open(scaler_path, "wb") as fh:
            pickle.dump(scaler, fh)

        print(
            f"{stage:<6} | "
            f"{threshold_cfg['T']:<6.3f} {threshold_cfg['W']:<4d} | "
            f"{threshold_cfg['validation_raw_exceedance_rate']:<7.4f} "
            f"{attack_metrics['precision']:<8.3f} "
            f"{attack_metrics['recall']:<7.3f} "
            f"{attack_metrics['f1']:<6.3f}"
        )

    stats_path = os.path.join(OUTPUT_DIR, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats_output, fh, indent=2)
    print(f"\nSaved stats -> {stats_path}")

    thresholds_path = os.path.join(OUTPUT_DIR, "thresholds.json")
    with open(thresholds_path, "w", encoding="utf-8") as fh:
        json.dump(threshold_output, fh, indent=2)
    print(f"Saved thresholds -> {thresholds_path}")

    print("\nExport complete.")


if __name__ == "__main__":
    main()
