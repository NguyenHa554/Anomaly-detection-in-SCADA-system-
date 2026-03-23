"""
alert_service.py — Creates and persists alerts when anomalies are detected.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
from sqlalchemy.orm import Session
from backend.database import Alert


COOLDOWN_SECONDS = 300  # 5-minute cooldown per stage


def get_severity(score: float) -> str:
    if score >= 2.0:   return "CRITICAL"
    if score >= 1.5:   return "HIGH"
    if score >= 1.2:   return "MEDIUM"
    return "LOW"


def build_message(stage: str, severity: str, detection: Dict) -> str:
    return (
        f"Anomaly detected in Stage {stage}. "
        f"Severity: {severity}. "
        f"Z-Score: {detection['max_z_score']:.2f} "
        f"(threshold: {detection['threshold']:.2f}). "
        f"Score: {detection['anomaly_score']:.2f}x threshold. "
        f"Possible cyberattack — investigate immediately."
    )


class AlertService:
    def __init__(self):
        # Track last alert time per stage to enforce cooldown
        self._last_alert: Dict[str, datetime] = {}

    def _in_cooldown(self, stage: str) -> bool:
        last = self._last_alert.get(stage)
        if last is None:
            return False
        return (datetime.now(timezone.utc) - last).total_seconds() < COOLDOWN_SECONDS

    def process(self, detection: Dict, db: Session) -> Optional[Dict]:
        """
        Given a detection result, decide whether to raise an alert.
        Saves to DB and returns the alert dict (or None if suppressed).
        """
        if not detection.get("is_anomaly"):
            return None

        stage = detection["stage"]

        if self._in_cooldown(stage):
            return None

        severity = get_severity(detection["anomaly_score"])
        message  = build_message(stage, severity, detection)
        now      = datetime.now(timezone.utc)

        alert = Alert(
            created_at    = now,
            stage         = stage,
            severity      = severity,
            anomaly_score = detection["anomaly_score"],
            max_z_score   = detection["max_z_score"],
            threshold     = detection["threshold"],
            message       = message,
            acknowledged  = False,
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)

        self._last_alert[stage] = now

        return {
            "id":            alert.id,
            "created_at":    now.isoformat(),
            "stage":         stage,
            "severity":      severity,
            "anomaly_score": detection["anomaly_score"],
            "max_z_score":   detection["max_z_score"],
            "threshold":     detection["threshold"],
            "message":       message,
            "acknowledged":  False,
        }


# Singleton
alert_service = AlertService()
