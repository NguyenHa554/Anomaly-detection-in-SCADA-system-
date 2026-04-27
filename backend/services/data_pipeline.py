"""
data_pipeline.py — Orchestrates: ingest → detect → alert → broadcast.
"""

import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional
from sqlalchemy.orm import Session

from backend.services.anomaly_detector import detector, STAGE_FEATURES
from backend.services.alert_service import alert_service, get_alert_type
from backend.database import SensorReading, Anomaly

STAGE_DISPLAY_FIELDS = {
    "P1": {
        "sensors": ["FIT 101", "LIT 101"],
        "actuators": ["MV 101", "P101 Status", "P102 Status"],
    },
    "P2": {
        "sensors": ["AIT 201", "AIT 202", "FIT 201"],
        "actuators": ["P203 Status", "MV201"],
    },
    "P3": {
        "sensors": ["AIT 301", "DPIT 301", "FIT 301", "LIT 301"],
        "actuators": ["P301 Status", "MV 301"],
    },
    "P4": {
        "sensors": ["AIT 401", "AIT 402", "FIT 401", "LIT 401", "LS 401"],
        "actuators": ["P401 Status", "P402 Status", "P403 Status", "P404 Status", "UV401"],
    },
    "P5": {
        "sensors": ["FIT 501", "PIT 501", "AIT 501"],
        "actuators": ["P501 Status", "MV 501"],
    },
    "P6": {
        "sensors": ["FIT 601", "LSH 601", "LSH 602", "LSH 603", "LSL 601", "LSL 602", "LSL 603"],
        "actuators": ["P601 Status", "P602 Status", "P603 Status"],
    },
}
ALL_FEATURES = tuple(
    dict.fromkeys(
        feature
        for cols in STAGE_FEATURES.values()
        for feature in cols
    )
)
DISPLAY_FIELDS = tuple(
    dict.fromkeys(
        field
        for stage_fields in STAGE_DISPLAY_FIELDS.values()
        for group in ("sensors", "actuators")
        for field in stage_fields[group]
    )
)
ALL_FEATURES = tuple(dict.fromkeys((*ALL_FEATURES, *DISPLAY_FIELDS)))
ALERTING_STAGES = tuple(stage for stage in STAGE_FEATURES if stage not in {"P1", "P6"})
LAST_KNOWN_VALUES: Dict[str, float] = {}
LAST_ANOMALY_STATES: Dict[str, bool] = {
    stage: False for stage in STAGE_FEATURES
}


def _coerce_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number):
        return None
    return number


def _normalise_row(row: Dict) -> Dict[str, float]:
    """
    Approximate notebook ffill() for streaming input by carrying forward
    the most recent valid value for each selected feature.
    """
    normalised: Dict[str, float] = {}

    for feature in ALL_FEATURES:
        current = _coerce_float(row.get(feature))
        if current is None:
            current = LAST_KNOWN_VALUES.get(feature, 0.0)
        else:
            LAST_KNOWN_VALUES[feature] = current
        normalised[feature] = current

    return normalised


def reset_runtime_state():
    LAST_KNOWN_VALUES.clear()
    for stage in LAST_ANOMALY_STATES:
        LAST_ANOMALY_STATES[stage] = False


async def process_row(row: Dict, db: Session, broadcast_fn) -> Dict:
    """
    Process one row of sensor data through the full pipeline.
    `row` is a flat dict of {feature_name: value, ...}.
    `broadcast_fn` is an async callable that pushes a message to all WebSocket clients.
    """
    timestamp = datetime.now(timezone.utc)
    stage_results = []
    feature_row = _normalise_row(row)

    for stage, cols in STAGE_FEATURES.items():
        # Build stage input using notebook-aligned feature values.
        features = np.array([feature_row[c] for c in cols], dtype=np.float32)

        detector.add_data_point(stage, features)
        result = detector.predict(stage)
        result["contributes_to_overall_alert"] = stage in ALERTING_STAGES
        result["timestamp"] = timestamp.isoformat()
        is_anomaly = bool(result.get("is_anomaly", False))
        is_episode_start = is_anomaly and not LAST_ANOMALY_STATES.get(stage, False)
        LAST_ANOMALY_STATES[stage] = is_anomaly
        result["is_episode_start"] = is_episode_start
        display_fields = STAGE_DISPLAY_FIELDS.get(stage, {"sensors": [], "actuators": []})
        sensor_values = {name: feature_row[name] for name in display_fields["sensors"] if name in feature_row}
        actuator_values = {name: feature_row[name] for name in display_fields["actuators"] if name in feature_row}
        raw_values = {name: feature_row[name] for name in cols if name in feature_row}
        result["sensor_values"] = sensor_values
        result["actuator_values"] = actuator_values

        # Persist sensor reading
        reading = SensorReading(
            timestamp       = timestamp,
            stage           = stage,
            z_score         = result.get("max_z_score"),
            is_anomaly      = is_anomaly,
            sensor_values   = sensor_values,
            actuator_values = actuator_values,
            raw_values      = raw_values,
        )
        db.add(reading)

        if result.get("status") == "ok" and is_episode_start:
            anomaly = Anomaly(
                timestamp     = timestamp,
                stage         = stage,
                anomaly_score = result["anomaly_score"],
                max_z_score   = result["max_z_score"],
                threshold     = result["threshold"],
                severity      = get_alert_type(True),
                details       = {
                    "sensor_values": sensor_values,
                    "actuator_values": actuator_values,
                    "raw_values": raw_values,
                },
            )
            db.add(anomaly)

        # Only production stages contribute to final alerts. P6 remains visible for analysis.
        if (
            result.get("status") == "ok"
            and is_episode_start
            and stage in ALERTING_STAGES
        ):
            alert_dict = alert_service.process(result, db)
            if alert_dict:
                await broadcast_fn({"type": "alert", "alert": alert_dict})

        stage_results.append(result)

    db.commit()

    # Ensemble: overall anomaly if any production stage fires.
    overall = any(
        r.get("is_anomaly", False) and r.get("contributes_to_overall_alert", False)
        for r in stage_results
    )

    payload = {
        "type":            "sensor_update",
        "timestamp":       timestamp.isoformat(),
        "stages":          stage_results,
        "overall_anomaly": overall,
        "raw_data":        feature_row,
    }

    await broadcast_fn(payload)
    return payload
