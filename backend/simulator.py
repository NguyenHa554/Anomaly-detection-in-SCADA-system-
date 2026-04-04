"""
simulator.py — Replays SWaT.csv through the API to demonstrate real-time detection.

Usage (separate terminal, while backend is running):
    python -m backend.simulator
"""

import asyncio, aiohttp, pandas as pd
from datetime import datetime

API_URL   = "http://localhost:8000"
CSV_FILE  = "SWaT.csv"
DELAY     = 0.2   # seconds between rows (0.2 s ≈ 5x real speed)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_FILE, header=1, low_memory=False)

    # Mirror export_models.py: treat the first column as time when there is
    # no explicit "time"/"timestamp" header and drop non-data rows instead of
    # coercing them into an all-zero sample.
    time_cands = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
    ts_col = time_cands[0] if time_cands else df.columns[0]
    parsed_ts = pd.to_datetime(df[ts_col], format="ISO8601", errors="coerce", utc=True)
    df = df.loc[parsed_ts.notna()].copy()

    # Encode Active/Inactive
    for col in df.columns:
        if df[col].astype(str).str.contains("Active|Inactive", case=False, na=False).any():
            df[col] = df[col].map({"Active": 1, "Inactive": 0})

    for col in df.columns:
        if col == ts_col:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.drop(columns=[ts_col])


def row_to_payload(row: pd.Series) -> dict:
    payload = {}
    for key, value in row.items():
        if pd.isna(value):
            payload[key] = None
        elif hasattr(value, "item"):
            payload[key] = value.item()
        else:
            payload[key] = value
    return payload


async def run():
    df = load_data()
    print(f"Loaded {len(df)} rows from {CSV_FILE}.")
    print("Streaming to API … (Ctrl+C to stop)\n")

    async with aiohttp.ClientSession() as session:
        for idx, (_, row) in enumerate(df.iterrows()):
            payload = row_to_payload(row)
            try:
                async with session.post(f"{API_URL}/api/ingest", json=payload) as resp:
                    if resp.status != 200:
                        print(f"  Row {idx}: API error {resp.status}")
            except Exception as e:
                print(f"  Row {idx}: {e}")
                await asyncio.sleep(2)
                continue

            if idx % 100 == 0:
                print(f"  Sent row {idx} / {len(df)}  [{datetime.now().strftime('%H:%M:%S')}]")

            await asyncio.sleep(DELAY)

    print("\nSimulation complete.")


if __name__ == "__main__":
    asyncio.run(run())
