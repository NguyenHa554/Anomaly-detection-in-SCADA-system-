"""
export_models.py — Re-trains 4 CNN models from data2.ipynb and saves:
  - ml_models/{P1,P2,P3,P5}_model.keras
  - ml_models/{P1,P2,P3,P5}_scaler.pkl
  - ml_models/stats.json         (mu / sigma per stage)
  - ml_models/thresholds.json    (best T / W from notebook grid-search)

Run once before starting the backend:
  python export_models.py
"""

import os, json, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Dropout,
    Input, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_FILE  = "SWaT.csv"
OUTPUT_DIR = "ml_models"
TIME_STEPS = 200

# Best T and W found by grid-search in the notebook
BEST_PARAMS = {
    "P1": {"T": 2.0, "W": 30},
    "P2": {"T": 2.0, "W": 300},
    "P3": {"T": 2.0, "W": 30},
    "P5": {"T": 2.0, "W": 30},
}

# 44 features selected by correlation with 'Attack' in the notebook
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
    "FIT 601", "LSH 601", "P601 Status",
]

# Attack periods from the notebook (UTC)
ATTACK_PERIODS_UTC = [
    ("2019-07-20 07:08:46", "2019-07-20 07:10:31"),
    ("2019-07-20 07:15:00", "2019-07-20 07:19:32"),
    ("2019-07-20 07:26:57", "2019-07-20 07:30:48"),
    ("2019-07-20 07:38:50", "2019-07-20 07:46:20"),
    ("2019-07-20 07:54:00", "2019-07-20 07:56:00"),
    ("2019-07-20 08:02:56", "2019-07-20 08:16:18"),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def add_derivative_features(data: np.ndarray) -> np.ndarray:
    diff = np.diff(data, axis=0, prepend=data[0].reshape(1, -1))
    return np.concatenate([data, diff], axis=1)


def create_sequences(X: np.ndarray, time_steps: int = 200):
    Xs, ys = [], []
    n_orig = X.shape[1] // 2      # original (non-derivative) feature count
    for i in range(len(X) - time_steps):
        Xs.append(X[i : i + time_steps])
        ys.append(X[i + time_steps, :n_orig])
    return np.array(Xs), np.array(ys)


def get_stage_features(columns) -> dict:
    stages = {}
    for num in [1, 2, 3, 5]:
        cols = [c for c in columns if str(num) in c or f"P{num}" in c]
        if cols:
            stages[f"P{num}"] = cols
    return stages


def build_model(time_steps: int, n_in: int, n_out: int) -> tf.keras.Model:
    return Sequential([
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
    ])


# ── Load & preprocess data ─────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_FILE, header=1, low_memory=False)

# Detect and parse timestamp column
time_cands = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
ts_col = time_cands[0] if time_cands else df.columns[0]
df = df[~df[ts_col].astype(str).str.lower().eq(ts_col.lower())]
df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
df = df.dropna(subset=[ts_col]).set_index(ts_col)

# Encode Active/Inactive status columns
for col in df.columns:
    if df[col].astype(str).str.contains("Active|Inactive", case=False, na=False).any():
        df[col] = df[col].map({"Active": 1, "Inactive": 0})

# Force numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Label attack windows
df["Attack"] = 0
for s, e in ATTACK_PERIODS_UTC:
    start = pd.to_datetime(s).tz_localize("UTC")
    end   = pd.to_datetime(e).tz_localize("UTC")
    df.loc[start:end, "Attack"] = 1

# Build modelling dataframe
df_model = df[SELECTED_FEATURES + ["Attack"]].copy()
df_model.ffill(inplace=True)
df_model.bfill(inplace=True)

# Train/test split (same cut as notebook)
split_ts = pd.to_datetime("2019-07-20 07:00:00").tz_localize("UTC")
df_train = df_model.loc[df_model.index <= split_ts]
df_test  = df_model.loc[df_model.index >  split_ts]

X_train = df_train[SELECTED_FEATURES]
X_test  = df_test[SELECTED_FEATURES]

stage_mapping = get_stage_features(X_train.columns)
print(f"Stages found: {list(stage_mapping.keys())}")

# ── Train, evaluate, and save each stage ──────────────────────────────────────
stats_output = {}

for stage, cols in stage_mapping.items():
    print(f"\n{'='*55}\n  Stage: {stage}  ({len(cols)} features)\n{'='*55}")

    # Scale
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(X_train[cols].values)
    test_scaled  = scaler.transform(X_test[cols].values)

    # Derivative features
    train_enh = add_derivative_features(train_scaled)
    test_enh  = add_derivative_features(test_scaled)

    # Sequences
    X_seq_tr, y_seq_tr = create_sequences(train_enh, TIME_STEPS)
    X_seq_te, y_seq_te = create_sequences(test_enh,  TIME_STEPS)

    n_in  = X_seq_tr.shape[2]
    n_out = y_seq_tr.shape[1]

    # Build & train
    model = build_model(TIME_STEPS, n_in, n_out)
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_seq_tr, y_seq_tr,
        epochs=100, batch_size=64,
        validation_split=0.1,
        callbacks=[es],
        verbose=1,
    )

    # Compute mu / sigma on train errors (used for z-score at inference time)
    tr_pred = model.predict(X_seq_tr, verbose=0)
    tr_err  = np.abs(tr_pred - y_seq_tr)
    mu      = np.mean(tr_err, axis=0)
    sigma   = np.std(tr_err,  axis=0)

    stats_output[stage] = {"mu": mu.tolist(), "sigma": sigma.tolist()}

    # Save model in native Keras format for better cross-version compatibility.
    model_path = os.path.join(OUTPUT_DIR, f"{stage}_model.keras")
    model.save(model_path)
    print(f"  ✓ Saved model  → {model_path}")

    # Save scaler
    scaler_path = os.path.join(OUTPUT_DIR, f"{stage}_scaler.pkl")
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)
    print(f"  ✓ Saved scaler → {scaler_path}")

# ── Save JSON artifacts ────────────────────────────────────────────────────────
stats_path = os.path.join(OUTPUT_DIR, "stats.json")
with open(stats_path, "w") as fh:
    json.dump(stats_output, fh, indent=2)
print(f"\n✓ Saved stats       → {stats_path}")

thresh_path = os.path.join(OUTPUT_DIR, "thresholds.json")
with open(thresh_path, "w") as fh:
    json.dump(BEST_PARAMS, fh, indent=2)
print(f"✓ Saved thresholds  → {thresh_path}")

print("\n🎉  Export complete!  All files are in", OUTPUT_DIR)
