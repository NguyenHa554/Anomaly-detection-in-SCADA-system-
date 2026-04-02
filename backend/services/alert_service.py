"""
alert_service.py - Creates and persists alerts when anomaly episodes begin.
"""

from datetime import datetime, timezone
from typing import Dict, Optional

from sqlalchemy.orm import Session

from backend.database import Alert

ALERT_TYPE_NORMAL = "NORMAL"
ALERT_TYPE_DANGER = "DANGER"


def get_alert_type(is_anomaly: bool) -> str:
    return ALERT_TYPE_DANGER if is_anomaly else ALERT_TYPE_NORMAL


def build_message(stage: str, alert_type: str, detection: Dict) -> str:
    return (
        f"Confirmed anomaly episode started in Stage {stage}. "
        f"Type: {alert_type}. "
        f"Z-Score: {detection['max_z_score']:.2f} "
        f"(threshold: {detection['threshold']:.2f}). "
        f"Score: {detection['anomaly_score']:.2f}x threshold after the full detection window was exceeded. "
        "Possible cyberattack - investigate immediately."
    )


class AlertService:
    def process(self, detection: Dict, db: Session) -> Optional[Dict]:
        """
        Given a detection result, decide whether to raise an alert.
        Saves to DB and returns the alert dict (or None if suppressed).
        """
        if not detection.get("is_episode_start"):
            return None

        stage = detection["stage"]
        alert_type = get_alert_type(True)
        message = build_message(stage, alert_type, detection)
        now = datetime.now(timezone.utc)

        alert = Alert(
            created_at=now,
            stage=stage,
            severity=alert_type,
            anomaly_score=detection["anomaly_score"],
            max_z_score=detection["max_z_score"],
            threshold=detection["threshold"],
            message=message,
            acknowledged=False,
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)

        return {
            "id": alert.id,
            "created_at": now.isoformat(),
            "stage": stage,
            "severity": alert_type,
            "anomaly_score": detection["anomaly_score"],
            "max_z_score": detection["max_z_score"],
            "threshold": detection["threshold"],
            "message": message,
            "acknowledged": False,
        }


alert_service = AlertService()
