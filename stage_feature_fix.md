# Fix: Stage Feature Assignment in SWaT Dataset

## Problem

The original `get_stage_features` function assigns features to stages incorrectly by checking whether the stage digit appears **anywhere** in the column name:

```python
# BUGGY VERSION
def get_stage_features(df):
    stages = {}
    all_cols = df.columns
    for stage_num in [1, 2, 3, 5]:
        cols = [c for c in all_cols if str(stage_num) in c or f"P{stage_num}" in c]
        if cols:
            stages[f'P{stage_num}'] = cols
    return stages
```

### Why This Is Wrong

In SWaT's naming convention, sensor/actuator IDs are 3-digit numbers where the **first digit encodes the stage**:

| Stage | Sensor number range | Examples |
|-------|---------------------|---------|
| P1 | 1xx | FIT 101, LIT 101, MV 101 |
| P2 | 2xx | AIT 201, FIT 201, MV201 |
| P3 | 3xx | AIT 301, LIT 301, MV 301 |
| P4 | 4xx | AIT 402, FIT 401, LIT 401 |
| P5 | 5xx | AIT 501, FIT 501, PIT 501 |
| P6 | 6xx | FIT 601, LSH 601 |

The buggy code checks `str(stage_num) in c`, which matches the digit **anywhere** in the string — not just the first digit of the sensor number. This causes features to be incorrectly shared across stages.

### Concrete Examples of Misassignment

| Column | Correct stage | Buggy code assigns to |
|--------|---------------|-----------------------|
| `AIT 501` | P5 | **P1 and P5** (`"1"` appears in `"501"`) |
| `FIT 501` | P5 | **P1 and P5** |
| `PIT 501` | P5 | **P1 and P5** |
| `AIT 201` | P2 | **P1 and P2** (`"1"` appears in `"201"`) |
| `FIT 201` | P2 | **P1 and P2** |
| `PIT 501` | P5 | **P1 and P5** |
| `AIT 502` | P5 | **P2 and P5** (`"2"` appears in `"502"`) |

This inflates the feature sets for certain stages and blurs the per-stage separation that the paper's ensemble method depends on.

---

## Fix

Use a regex that matches only when the stage digit is the **first digit of the 3-digit sensor number** (e.g., `101`, `201`, `501`):

```python
import re

def get_stage_features(df):
    stages = {}
    all_cols = df.columns
    for stage_num in [1, 2, 3, 5]:
        cols = [
            c for c in all_cols
            if re.search(rf'\b{stage_num}\d{{2}}\b', c)   # matches 1xx, 2xx, etc.
            or re.fullmatch(rf'P{stage_num}_STATE', c)     # matches P1_STATE, P2_STATE, etc.
        ]
        if cols:
            stages[f'P{stage_num}'] = cols
    return stages
```

### How the Regex Works

- `\b{stage_num}\d{2}\b` — word boundary + stage digit + exactly 2 more digits + word boundary
  - `FIT 101` → `101` matches `\b1\d{2}\b` ✅ → P1
  - `AIT 501` → `501` matches `\b5\d{2}\b` ✅ → P5 only (not P1)
  - `FIT 201` → `201` matches `\b2\d{2}\b` ✅ → P2 only (not P1)
- `P{stage_num}_STATE` — exact match for state columns like `P1_STATE`, `P3_STATE`

---

## Impact on the Paper's Method

The paper (Kravchik & Shabtai, 2018) trains a **separate 1D CNN model per process stage** and then takes an ensemble of their predictions. This approach only works correctly if each stage model sees exclusively its own sensors. With the buggy assignment:

- P1's model receives features from P5 sensors — learning spurious cross-stage correlations.
- The ensemble loses its per-stage independence.
- Attack detection performance may be degraded or misleadingly inflated.

The fix ensures each stage model is trained on the correct, non-overlapping set of features, consistent with the paper's design.

---

## Verification

After applying the fix, you can verify the assignments are correct and non-overlapping:

```python
stage_mapping = get_stage_features(X_train)

# Print features per stage
for stage, cols in stage_mapping.items():
    print(f"\n{stage} ({len(cols)} features):")
    for c in cols:
        print(f"  {c}")

# Check for overlaps between stages
all_assigned = []
for stage, cols in stage_mapping.items():
    overlap = set(cols) & set(all_assigned)
    if overlap:
        print(f"WARNING: {stage} overlaps with a previous stage: {overlap}")
    all_assigned.extend(cols)

print("\nNo overlaps found." if len(all_assigned) == len(set(all_assigned)) else "Overlaps detected!")
```
