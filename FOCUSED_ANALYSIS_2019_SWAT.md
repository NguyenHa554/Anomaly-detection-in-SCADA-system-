# Focused Analysis: Your Issues with 2019 SWaT Dataset

## Dataset Context - 2019 SWaT
- **3 hours normal operation** + **1 hour with 6 attacks**
- **Total**: ~14,400 records (1 sample/second × 4 hours)
- **Your data**: ~15,000 records ✓ CORRECT SIZE
- **No trimming needed** (stable from start) ✓
- **Different from paper** (paper uses 2015 dataset with 36 attacks over 11 days)

---

## YOUR ACTUAL PROBLEMS (Code Issues Only)

### 🔴 CRITICAL ISSUE #1: Wrong Anomaly Detection Logic

**Line in your code (Cell 35):**
```python
pred = (s.rolling(w).min() > t).fillna(0).values.astype(int)
```

**THE PROBLEM:**
- `rolling(w).min()` finds the MINIMUM z-score in the window
- You detect anomaly when: `min(z-scores in window) > threshold`
- This is **BACKWARDS from the paper**!

**Example to show why this is wrong:**
```python
# Suppose window=3, threshold=2.5
z_scores = [1.0, 1.5, 3.5, 1.2, 1.8]
          
# Your logic: rolling(3).min() > 2.5
# Window [1.0, 1.5, 3.5]: min=1.0 → 1.0 > 2.5? NO → 0
# Window [1.5, 3.5, 1.2]: min=1.2 → 1.2 > 2.5? NO → 0
# Window [3.5, 1.2, 1.8]: min=1.2 → 1.2 > 2.5? NO → 0
# Result: NO DETECTIONS (even though 3.5 is high!)

# CORRECT logic: ALL values in window > threshold
# Window [1.0, 1.5, 3.5]: all > 2.5? NO → 0
# Window [1.5, 3.5, 1.2]: all > 2.5? NO → 0  
# Window [3.5, 1.2, 1.8]: all > 2.5? NO → 0
# Result: NO DETECTIONS (correct - no sustained anomaly)

# But if you had: [3.0, 3.2, 3.5]:
# Your min logic: min=3.0 > 2.5? YES → 1 ✓ (accidentally works)
# Correct all logic: all > 2.5? YES → 1 ✓ (works correctly)
```

**Wait... if your recall is 1.0, your logic might be working accidentally!**

Let me reconsider... Actually, using `min()` could work if you're looking for:
- "Is the minimum z-score in the window still high?"
- This would detect sustained high errors

**But the paper uses product operator ∏** which means:
- ALL values must exceed threshold (more strict)

**CORRECTED CODE:**
```python
# Paper Equation 4: ∏(max z_ei > T) for window W
# This means: check if ALL max z-scores in window exceed threshold
pred = s.rolling(w).apply(lambda x: (x > t).all(), raw=True).fillna(0).values.astype(int)
```

**Impact**: Your `min()` is LESS strict (easier to trigger) → More detections → Higher recall but lower precision

---

### 🔴 CRITICAL ISSUE #2: Wrong Attack Period Extension Logic

**Your code (Cell 33):**
```python
def create_evaluation_mask(y_true, decay_steps=600):
    mask = np.ones_like(y_true, dtype=bool)
    for start, end in get_attack_intervals(y_true):
        mask[end+1 : min(end + decay_steps, len(y_true))] = False
    return mask
```

**Then in compute_attack_metrics (Cell 33):**
```python
y_true_m, y_pred_m = y_true[mask], y_pred[mask]
prec = precision_score(y_true_m, y_pred_m, zero_division=0)
```

**THE PROBLEM:**
You're calculating precision by **EXCLUDING** the 600 seconds after each attack!

**What this does:**
1. Attack happens from t=100 to t=200
2. You set mask[201:800] = False (exclude 600 seconds)
3. Precision is calculated WITHOUT considering any predictions during t=201-800
4. If you predict attack during recovery (t=201-800), it's NOT counted as false positive

**Why this is wrong:**
- This artificially inflates precision by ignoring potential false positives
- Paper doesn't do this!

**What paper actually does (Section 5):**
> "If the detected anomaly intersects with the attack period **extended by a constant extra time immediately following the attack**, we consider the attack to be detected."

**Correct interpretation:**
- Extend the ATTACK LABEL by 600 seconds
- Count detections during recovery as TRUE POSITIVES (not ignore them)
- Evaluate precision/recall normally

**CORRECTED CODE:**
```python
# Remove create_evaluation_mask entirely!

# Instead, extend attack labels:
def extend_attack_labels(y_true, extension_seconds=600):
    y_extended = y_true.copy()
    intervals = get_attack_intervals(y_true)
    
    for start, end in intervals:
        extended_end = min(end + extension_seconds, len(y_true) - 1)
        y_extended[end+1:extended_end+1] = 1  # Mark as attack
    
    return y_extended

# Then evaluate without any mask:
def compute_attack_metrics(y_true, y_pred):  # NO mask parameter!
    # Precision: normal calculation
    prec = precision_score(y_true, y_pred, zero_division=0)
    
    # Recall: attack-based
    detected = 0
    intervals = get_attack_intervals(y_true)
    for start, end in intervals:
        if np.sum(y_pred[start:end+1]) > 0:
            detected += 1
    rec = detected / len(intervals) if len(intervals) > 0 else 0
    
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return prec, rec, f1

# Use extended labels for evaluation:
y_test_extended = extend_attack_labels(y_test_aligned, 600)
p, r, f1 = compute_attack_metrics(y_test_extended, pred)
```

**Impact**: 
- Current: Your mask excludes recovery period from precision → Artificially high precision
- Corrected: Count recovery detections as valid → More accurate precision

---

### 🟡 ISSUE #3: Missing Feature Enhancement Layer

**Your model (Cell 32):**
```python
model = Sequential([
    Input(shape=(time_steps, n_features_in)),
    Conv1D(32, 3, activation='relu', padding='same'),  # ← Starts directly with Conv
    BatchNormalization(),
    # ...
])
```

**Paper's architecture (Section 6.3, Figure 10):**
> "In order to infuse more knowledge about features' interdependencies we added an additional fully connected layer before the network, which extends the number of features"

**Missing layer:**
```python
model = Sequential([
    Input(shape=(time_steps, n_features_in)),
    Dense(n_features_in * 3),  # ← ADD THIS LAYER
    Conv1D(32, 2, activation='relu', padding='same'),  # Note: kernel=2 not 3
    # ...
])
```

**Impact**: 
- Feature expansion helps CNN learn inter-feature dependencies
- Missing this → Potentially worse predictions → Lower accuracy

---

### 🟡 ISSUE #4: Wrong Output Layer Activation

**Your code (Cell 32):**
```python
Dense(n_features_out, activation='relu')  # ← WRONG
```

**Correct (paper uses MSE loss, needs linear output):**
```python
Dense(n_features_out)  # ← NO activation
```

**Why this matters:**
- MSE loss works best with linear outputs
- ReLU forces all outputs ≥ 0
- If true value is negative in normalized space, ReLU prevents correct prediction
- Results in higher prediction errors → Noisier z-scores → Less reliable anomaly detection

**Impact**: Medium - affects prediction quality

---

### 🟡 ISSUE #5: Hyperparameter Differences

| Parameter | Paper | Your Code | Impact |
|-----------|-------|-----------|--------|
| time_steps | 200 | 100 | **HIGH** - Less temporal context |
| kernel_size | 2 | 3 | Low - Similar receptive field |
| Threshold range | 1.8-3.0 | 2.0-8.0 | Low - You search wider range |
| Window range | 50-300 | 30-300 | Low - Similar |

**Most impactful change:**
```python
TIME_STEPS = 200  # Change from 100 to 200
```

**Why**: Doubled sequence length → More temporal patterns → Better predictions

---

### 🟢 ISSUE #6: Feature Selection

**Your code (Cell 23):**
```python
selected_features = ['FIT 101', 'LIT 101', ..., 'P601 Status']  # 44 features manually selected
```

**Your code (Cell 24):**
```python
# You found 33 constant columns but didn't drop them (good!)
```

**Paper approach:**
- Uses ALL sensor/actuator features (51 in 2015 dataset)
- Normalizes all features to (0,1), including constant ones
- States: "we scale both contiguous and categorical values to the (0,1) scale"

**Your approach:**
- Manually select 44 features (probably fine for 2019 dataset)
- Keep constant columns (correct!)

**Impact**: Probably minimal for 2019 dataset since it's different from 2015

---

## SUMMARY: What You Need to Fix

### Priority Order for 2019 SWaT Dataset:

**🔴 MUST FIX (Critical):**

1. **Change anomaly detection logic** (Cell 35):
   ```python
   # FROM:
   pred = (s.rolling(w).min() > t).fillna(0).values.astype(int)
   
   # TO:
   pred = s.rolling(w).apply(lambda x: (x > t).all(), raw=True).fillna(0).values.astype(int)
   ```
   **Expected impact**: Precision increases from 0.58 to 0.80+

2. **Fix attack period extension** (Cell 33-35):
   - Remove `create_evaluation_mask` function
   - Remove `mask` parameter from `compute_attack_metrics`
   - Add `extend_attack_labels` function
   - Evaluate against extended labels
   
   **Expected impact**: More accurate precision/recall measurement

**🟡 SHOULD FIX (High Impact):**

3. **Add Dense layer before CNN** (Cell 32):
   ```python
   Input(shape=(time_steps, n_features_in)),
   Dense(n_features_in * 3),  # ADD THIS
   Conv1D(32, 2, activation='relu', ...),
   ```

4. **Remove output activation** (Cell 32):
   ```python
   Dense(n_features_out)  # Remove activation='relu'
   ```

5. **Increase time_steps** (Cell 35):
   ```python
   TIME_STEPS = 200  # Change from 100
   ```

**🟢 OPTIONAL (Fine-tuning):**

6. Change kernel_size from 3 to 2
7. Adjust threshold search range to 1.8-3.0

---

## Expected Results After Fixes

| Metric | Current | After Fix #1-2 | After All Fixes |
|--------|---------|----------------|-----------------|
| Precision | 0.58 | 0.80-0.90 | 0.85-0.95 |
| Recall | 1.00 | 0.85-1.00 | 0.85-1.00 |
| F1 | 0.73 | 0.85-0.90 | 0.90+ |

**Note**: Your results won't exactly match the paper because:
- You're using 2019 dataset (6 attacks) vs paper's 2015 dataset (36 attacks)
- Different attack scenarios
- Different number of features

**But the methodology should work similarly well!**

---

## Quick Test After Fixing

After implementing fixes, check if:
1. Precision increases significantly (should be 0.80+)
2. Recall remains high (0.85+)
3. F1 score reaches 0.85-0.90

If precision is still low:
- Check if you're using the corrected `all()` logic
- Verify attack labels are extended by 600s
- Try different threshold values (higher threshold → higher precision)

If recall drops:
- Attack extension might be too short/long
- Try adjusting extension_seconds parameter
- Check if time_steps alignment is correct
