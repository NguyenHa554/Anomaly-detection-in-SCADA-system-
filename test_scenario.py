"""
test_scenario.py — Validates that the anomaly detector correctly identifies
the 6 known attack periods from data2.ipynb and matches the notebook's
per-stage and overall evaluation results.

Notebook ground-truth
─────────────────────
Attack periods (UTC):
  A1  2019-07-20 07:08:46 – 07:10:31
  A2  2019-07-20 07:15:00 – 07:19:32
  A3  2019-07-20 07:26:57 – 07:30:48
  A4  2019-07-20 07:38:50 – 07:46:20
  A5  2019-07-20 07:54:00 – 07:56:00
  A6  2019-07-20 08:02:56 – 08:16:18

  Labels extended +600 s after each attack end (paper Section 5).

Per-stage results (notebook output):
  P1: Prec=0.80  Rec=0.95  F1=0.87   (T=2.0, W=30)
  P2: Prec=0.85  Rec=1.00  F1=0.92   (T=2.0, W=300)
  P3: Prec=0.81  Rec=0.93  F1=0.87   (T=2.0, W=30)
  P5: Prec=0.81  Rec=0.99  F1=0.89   (T=2.0, W=30)

  Final (OR-ensemble): Prec=0.8061  Rec=1.0000  F1=0.8926

Usage
─────
  # From the project root (c:\\nguyen\\ĐA KTMT):
  python test_scenario.py

Requirements: SWaT.csv, ml_models/ with .keras / .h5 + .pkl + stats.json + thresholds.json
"""

import json
import os
import pickle
import sys
import textwrap

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

# ─── Tolerance for metric comparison ─────────────────────────────────────────
METRIC_TOL = 0.05   # ±0.05 is considered "within tolerance" of notebook value

# ─── Project paths ────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
CSV_FILE  = os.path.join(ROOT, "SWaT.csv")
ML_DIR    = os.path.join(ROOT, "ml_models")

# ─── Define the 6 attack periods (UTC) ───────────────────────────────────────
ATTACK_PERIODS = [
    ("2019-07-20 07:08:46", "2019-07-20 07:10:31"),   # Attack 1
    ("2019-07-20 07:15:00", "2019-07-20 07:19:32"),   # Attack 2
    ("2019-07-20 07:26:57", "2019-07-20 07:30:48"),   # Attack 3
    ("2019-07-20 07:38:50", "2019-07-20 07:46:20"),   # Attack 4
    ("2019-07-20 07:54:00", "2019-07-20 07:56:00"),   # Attack 5
    ("2019-07-20 08:02:56", "2019-07-20 08:16:18"),   # Attack 6
]

# ─── Stage features (mirrors anomaly_detector.py exactly) ────────────────────
STAGE_FEATURES = {
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

# ─── Notebook expected metrics (for assertion) ────────────────────────────────
EXPECTED_PER_STAGE = {
    "P1": {"prec": 0.80, "rec": 0.95, "f1": 0.87},
    "P2": {"prec": 0.85, "rec": 1.00, "f1": 0.92},
    "P3": {"prec": 0.81, "rec": 0.93, "f1": 0.87},
    "P5": {"prec": 0.81, "rec": 0.99, "f1": 0.89},
}
EXPECTED_FINAL = {"prec": 0.8061, "rec": 1.0000, "f1": 0.8926}


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions (mirror data2.ipynb exactly)
# ═══════════════════════════════════════════════════════════════════════════════

def add_derivative_features(data: np.ndarray) -> np.ndarray:
    """Concatenate diff features — mirrors notebook exactly."""
    diff = np.diff(data, axis=0, prepend=data[0].reshape(1, -1))
    return np.concatenate([data, diff], axis=1)


def create_sequences(X: np.ndarray, time_steps: int = 200):
    """
    Build overlapping windows of length `time_steps`.
    Returns Xs  shape (N, time_steps, features)
            ys  shape (N, original_features)  — the next-step ground truth
    Mirrors create_sequences() in the notebook.
    """
    Xs, ys = [], []
    original_features_count = X.shape[1] // 2
    for i in range(len(X) - time_steps):
        Xs.append(X[i: i + time_steps])
        ys.append(X[i + time_steps, :original_features_count])
    return np.array(Xs), np.array(ys)


def get_attack_intervals(y_true: np.ndarray):
    diff   = np.diff(y_true, prepend=0)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0] - 1
    if len(starts) > len(ends):
        ends = np.append(ends, len(y_true) - 1)
    return list(zip(starts, ends))


def extend_attack_labels(y_true: np.ndarray, extension_seconds: int = 600) -> np.ndarray:
    """Extend attack labels by `extension_seconds` — mirrors notebook Section 5."""
    y_ext = y_true.copy()
    for start, end in get_attack_intervals(y_true):
        ext_end = min(end + extension_seconds, len(y_true) - 1)
        y_ext[end + 1: ext_end + 1] = 1
    return y_ext


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_swat() -> pd.DataFrame:
    print(f"[LOAD] Reading {CSV_FILE} …", flush=True)
    df = pd.read_csv(CSV_FILE, header=1, low_memory=False)

    # Identify timestamp column
    candidates = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
    ts_col = candidates[0] if candidates else df.columns[0]

    # Remove duplicate header rows that sometimes appear in SWaT exports
    df = df[~df[ts_col].astype(str).str.lower().eq(ts_col.lower())]

    # Parse timestamp
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True, infer_datetime_format=True)
    df = df.dropna(subset=[ts_col]).set_index(ts_col)

    # Encode Active / Inactive status columns
    for col in df.columns:
        if df[col].astype(str).str.contains("Active|Inactive", case=False, na=False).any():
            df[col] = df[col].map({"Active": 1, "Inactive": 0})

    # Coerce everything else to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"[LOAD] Total rows: {len(df)}  |  Columns: {len(df.columns)}", flush=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Label Building
# ═══════════════════════════════════════════════════════════════════════════════

def build_labels(df: pd.DataFrame) -> pd.Series:
    """Add Attack column using the 6 known attack periods."""
    attack = pd.Series(0, index=df.index, name="Attack")
    for start_str, end_str in ATTACK_PERIODS:
        s = pd.to_datetime(start_str).tz_localize("UTC")
        e = pd.to_datetime(end_str).tz_localize("UTC")
        attack[s:e] = 1
    total = int(attack.sum())
    print(f"[LABEL] Attack samples: {total}  "
          f"({100 * total / len(attack):.2f} % of all rows)", flush=True)
    return attack


# ═══════════════════════════════════════════════════════════════════════════════
# Stage Inference (offline batch — mirrors notebook Cell 16)
# ═══════════════════════════════════════════════════════════════════════════════

def run_stage(stage: str, X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame,
              TIME_STEPS: int = 200):
    """
    Run one stage through the saved model and return (test_err, mu, sigma).
    Loads the saved scaler and model from ml_models/ instead of re-training.
    """
    scaler_path = os.path.join(ML_DIR, f"{stage}_scaler.pkl")
    model_path  = os.path.join(ML_DIR, f"{stage}_model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(ML_DIR, f"{stage}_model.h5")
    stats_path  = os.path.join(ML_DIR, "stats.json")

    with open(scaler_path, "rb") as fh:
        scaler: MinMaxScaler = pickle.load(fh)
    with open(stats_path) as fh:
        stats = json.load(fh)

    model = tf.keras.models.load_model(model_path, compile=False)

    mu    = np.array(stats[stage]["mu"])
    sigma = np.array(stats[stage]["sigma"])

    # Scale using the fitted scaler (fitted on train during export)
    test_scaled = scaler.transform(X_test_raw.values)

    # Derivative features
    test_enhanced = add_derivative_features(test_scaled)

    # Create sequences
    X_seq_test, y_seq_test = create_sequences(test_enhanced, TIME_STEPS)

    # Predict
    test_pred = model.predict(X_seq_test, verbose=0)

    # MAE errors
    test_err = np.abs(test_pred - y_seq_test)

    return test_err, mu, sigma


# ═══════════════════════════════════════════════════════════════════════════════
# Detection Logic (mirrors notebook Cell 16 grid-search evaluate)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_stage(test_err: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                 T: float, W: int) -> np.ndarray:
    """
    Apply Z-score + rolling window to produce binary predictions.
    Equation 4 from paper: anomaly=1 only when all W consecutive z > T.
    """
    z     = (test_err - mu) / (sigma + 1e-8)
    max_z = np.max(z, axis=1)
    s     = pd.Series(max_z)
    pred  = (
        s.rolling(W)
         .apply(lambda x: 1 if (x > T).all() else 0, raw=True)
         .fillna(0)
         .values
         .astype(int)
    )
    return pred


# ═══════════════════════════════════════════════════════════════════════════════
# Assertion helper
# ═══════════════════════════════════════════════════════════════════════════════

def _check(name: str, computed: float, expected: float, tol: float = METRIC_TOL) -> bool:
    ok = abs(computed - expected) <= tol
    symbol = "✓" if ok else "✗"
    status = "PASS" if ok else "FAIL"
    print(f"    [{symbol}] {name}: computed={computed:.4f}  expected≈{expected:.4f}  "
          f"tol=±{tol:.2f}  → {status}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# Main Test Scenario
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  SCADA ANOMALY DETECTION — TEST SCENARIO")
    print("  Validates: attack detection + metric parity with data2.ipynb")
    print("=" * 70)

    # ── Step 1: Load and pre-process data ─────────────────────────────────────
    print("\n[STEP 1] Load SWaT dataset")
    df = load_swat()

    # ── Step 2: Build attack labels ───────────────────────────────────────────
    print("\n[STEP 2] Build ground-truth attack labels")
    attack = build_labels(df)
    df["Attack"] = attack

    # ── Step 3: Split exactly as notebook ─────────────────────────────────────
    print("\n[STEP 3] Time-based train/test split at 2019-07-20 07:00:00 UTC")
    ALL_FEATURES = []
    for feats in STAGE_FEATURES.values():
        for f in feats:
            if f not in ALL_FEATURES:
                ALL_FEATURES.append(f)

    split_ts = pd.to_datetime("2019-07-20 07:00:00").tz_localize("UTC")
    df_train  = df.loc[df.index <= split_ts]
    df_test   = df.loc[df.index >  split_ts]

    print(f"  Train: {len(df_train)} rows  |  Test: {len(df_test)} rows")
    print(f"  Test attack samples  : {int(df_test['Attack'].sum())}  "
          f"(expected ≈ 1981)")

    assert len(df_test) > 0, "Test set is empty — check CSV timestamp format."

    X_train = df_train[ALL_FEATURES].copy()
    X_test  = df_test[ALL_FEATURES].copy()
    y_test  = df_test["Attack"].values

    TIME_STEPS = 200
    # Align labels to sequence output (notebook: y_test_aligned = y_test.iloc[TIME_STEPS:])
    y_test_aligned  = y_test[TIME_STEPS:]
    y_test_extended = extend_attack_labels(y_test_aligned, extension_seconds=600)

    print(f"\n  Test samples (after alignment): {len(y_test_aligned)}")
    print(f"  Attack samples (extended):      {int(y_test_extended.sum())}")

    # ── Step 4: Load thresholds ───────────────────────────────────────────────
    print("\n[STEP 4] Load thresholds from ml_models/thresholds.json")
    with open(os.path.join(ML_DIR, "thresholds.json")) as fh:
        thresholds = json.load(fh)
    for stage, cfg in thresholds.items():
        print(f"  {stage}: T={cfg['T']}  W={cfg['W']}")

    # ── Step 5: Per-stage inference + evaluation ──────────────────────────────
    print("\n[STEP 5] Per-stage inference using saved models\n")
    all_passed  = True
    stage_preds = {}

    for stage, feats in STAGE_FEATURES.items():
        print(f"  ── Stage {stage} ({len(feats)} features) ──")

        # Filter to features this stage uses
        X_train_stage = X_train[feats].ffill().bfill().fillna(0)
        X_test_stage  = X_test[feats].ffill().bfill().fillna(0)

        test_err, mu, sigma = run_stage(stage, X_train_stage, X_test_stage, TIME_STEPS)

        T = thresholds[stage]["T"]
        W = thresholds[stage]["W"]
        pred = detect_stage(test_err, mu, sigma, T, W)

        stage_preds[stage] = pred

        # Compute metrics against extended labels
        prec = precision_score(y_test_extended, pred, zero_division=0)
        rec  = recall_score(y_test_extended, pred, zero_division=0)
        f1   = f1_score(y_test_extended, pred, zero_division=0)

        exp = EXPECTED_PER_STAGE[stage]
        p_ok  = _check("Precision", prec, exp["prec"])
        r_ok  = _check("Recall",    rec,  exp["rec"])
        f_ok  = _check("F1",        f1,   exp["f1"])

        if not (p_ok and r_ok and f_ok):
            all_passed = False

        # Per-attack detection report
        print(f"\n  Attack-period detection (stage {stage}):")
        for i, (start_str, end_str) in enumerate(ATTACK_PERIODS, 1):
            s = pd.to_datetime(start_str).tz_localize("UTC")
            e = pd.to_datetime(end_str).tz_localize("UTC")
            # Map to test-set index positions
            test_index = df_test.index
            mask = (test_index >= s) & (test_index <= e)
            pos_in_aligned = np.where(mask)[0]
            # Shift by TIME_STEPS (sequence alignment)
            pos_in_aligned = pos_in_aligned[pos_in_aligned >= TIME_STEPS] - TIME_STEPS
            if len(pos_in_aligned) == 0:
                print(f"    Attack {i} [{start_str}–{end_str}]: out of range / no aligned samples")
                continue
            detected_steps = int(pred[pos_in_aligned].sum())
            total_steps    = len(pos_in_aligned)
            flag = "✓ DETECTED" if detected_steps > 0 else "✗ MISSED"
            print(f"    Attack {i} [{start_str}–{end_str}]: "
                  f"{detected_steps}/{total_steps} steps detected  {flag}")
        print()

    # ── Step 6: Ensemble (OR across stages) ──────────────────────────────────
    print("[STEP 6] OR-ensemble of all stage predictions")
    final_pred = np.zeros_like(y_test_aligned, dtype=int)
    for preds in stage_preds.values():
        final_pred = np.bitwise_or(final_pred, preds)

    fin_p  = precision_score(y_test_extended, final_pred, zero_division=0)
    fin_r  = recall_score(y_test_extended, final_pred, zero_division=0)
    fin_f1 = f1_score(y_test_extended, final_pred, zero_division=0)

    print(f"\n  Ensemble metrics (extended labels):")
    fp_ok = _check("Precision", fin_p,  EXPECTED_FINAL["prec"])
    fr_ok = _check("Recall",    fin_r,  EXPECTED_FINAL["rec"])
    ff_ok = _check("F1",        fin_f1, EXPECTED_FINAL["f1"])
    if not (fp_ok and fr_ok and ff_ok):
        all_passed = False

    # ── Step 7: Per-attack summary ────────────────────────────────────────────
    print("\n[STEP 7] Final ensemble — per-attack detection summary")
    all_attacks_detected = True
    for i, (start_str, end_str) in enumerate(ATTACK_PERIODS, 1):
        s = pd.to_datetime(start_str).tz_localize("UTC")
        e = pd.to_datetime(end_str).tz_localize("UTC")
        test_index = df_test.index
        mask = (test_index >= s) & (test_index <= e)
        pos = np.where(mask)[0]
        pos = pos[pos >= TIME_STEPS] - TIME_STEPS
        if len(pos) == 0:
            print(f"  Attack {i}: [{start_str} → {end_str}] — no aligned samples found")
            continue
        detected = int(final_pred[pos].sum())
        total    = len(pos)
        flag     = "✓ DETECTED" if detected > 0 else "✗ MISSED"
        if detected == 0:
            all_attacks_detected = False
        first_detected_pos = pos[np.where(final_pred[pos] == 1)[0][0]] if detected > 0 else None
        if first_detected_pos is not None:
            first_detected_ts = df_test.index[first_detected_pos + TIME_STEPS]
            delay = (first_detected_ts - s).total_seconds()
            print(f"  Attack {i}: [{start_str} → {end_str}]  "
                  f"{detected}/{total} steps  {flag}  "
                  f"| First alert at {first_detected_ts.strftime('%H:%M:%S')} "
                  f"(+{delay:.0f}s after start)")
        else:
            print(f"  Attack {i}: [{start_str} → {end_str}]  0/{total} steps  {flag}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TEST RESULTS SUMMARY")
    print("=" * 70)

    checks = {
        "All 6 attacks detected by ensemble":   all_attacks_detected,
        "Ensemble metrics within tolerance":     fp_ok and fr_ok and ff_ok,
        "Per-stage metrics within tolerance":    all_passed,
    }

    overall_pass = True
    for desc, ok in checks.items():
        symbol = "✓ PASS" if ok else "✗ FAIL"
        print(f"  [{symbol}]  {desc}")
        if not ok:
            overall_pass = False

    print()
    if overall_pass:
        print("  ✅  ALL TESTS PASSED — detector behaviour matches notebook results.")
    else:
        print("  ❌  SOME TESTS FAILED — see details above.")
    print("=" * 70)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
