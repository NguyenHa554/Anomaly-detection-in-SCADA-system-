"""
data_pipeline.py — Orchestrates: ingest → detect → alert → broadcast.
"""

import numpy as np
from datetime import datetime, timezone
from typing import Dict, List
from sqlalchemy.orm import Session

from backend.services.anomaly_detector import detector, STAGE_FEATURES
from backend.services.alert_service import alert_service
from backend.database import SensorReading


async def process_row(row: Dict, db: Session, broadcast_fn) -> Dict:
    """
    Process one row of sensor data through the full pipeline.
    `row` is a flat dict of {feature_name: value, ...}.
    `broadcast_fn` is an async callable that pushes a message to all WebSocket clients.
    """
    timestamp = datetime.now(timezone.utc)
    stage_results = []

    for stage, cols in STAGE_FEATURES.items():
        # Extract features for this stage (fill 0 for missing columns)
        features = np.array([float(row.get(c, 0.0)) for c in cols], dtype=np.float32)

        detector.add_data_point(stage, features)
        result = detector.predict(stage)
        result["timestamp"] = timestamp.isoformat()

        # Persist sensor reading
        reading = SensorReading(
            timestamp  = timestamp,
            stage      = stage,
            z_score    = result.get("max_z_score"),
            is_anomaly = result.get("is_anomaly", False),
        )
        db.add(reading)

        # Maybe generate an alert
        if result.get("status") == "ok" and result.get("is_anomaly"):
            alert_dict = alert_service.process(result, db)
            if alert_dict:
                await broadcast_fn({"type": "alert", "data": alert_dict})

        stage_results.append(result)

    db.commit()

    # Ensemble: overall anomaly if ANY stage fires
    overall = any(r.get("is_anomaly", False) for r in stage_results)

    payload = {
        "type":            "sensor_update",
        "timestamp":       timestamp.isoformat(),
        "stages":          stage_results,
        "overall_anomaly": overall,
        "raw_data":        row, # Send raw data to frontend for display
    }

    await broadcast_fn(payload)
    return payload
