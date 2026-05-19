"""
incident_service.py - Groups stage anomaly episode starts into operator incidents.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from sqlalchemy.orm import Session

from backend.database import Alert, Incident
from backend.services.alert_service import ALERT_TYPE_DANGER
from backend.utils.time import utc_isoformat

INCIDENT_STATUS_OPEN = "OPEN"
INCIDENT_STATUS_CLOSED = "CLOSED"
INCIDENT_COOLDOWN = timedelta(minutes=5)


def _parse_timestamp(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _stage_evidence(detection: Dict) -> Dict:
    timestamp = _parse_timestamp(detection.get("timestamp")).isoformat()
    return {
        "max_z_score": detection["max_z_score"],
        "threshold": detection["threshold"],
        "anomaly_score": detection["anomaly_score"],
        "first_seen": timestamp,
        "last_seen": timestamp,
        "sensor_values": detection.get("sensor_values", {}),
        "actuator_values": detection.get("actuator_values", {}),
        "raw_values": detection.get("raw_values", {}),
    }


def _build_message(incident: Incident) -> str:
    affected = ", ".join(incident.affected_stages or [])
    return (
        f"Incident #{incident.id}: confirmed anomaly episode. "
        f"Primary stage: {incident.primary_stage}. "
        f"First detected: {incident.first_detected_stage}. "
        f"Affected stages: {affected}. "
        f"Max Z-Score: {incident.max_z_score:.2f}."
    )


def incident_to_dict(incident: Incident) -> Dict:
    message = _build_message(incident)
    return {
        "id": incident.id,
        "incident_id": incident.id,
        "created_at": utc_isoformat(incident.created_at),
        "updated_at": utc_isoformat(incident.updated_at),
        "start_time": utc_isoformat(incident.start_time),
        "end_time": utc_isoformat(incident.end_time),
        "stage": incident.primary_stage,
        "first_detected_stage": incident.first_detected_stage,
        "primary_stage": incident.primary_stage,
        "affected_stages": incident.affected_stages or [],
        "status": incident.status,
        "severity": incident.severity,
        "anomaly_score": incident.anomaly_score,
        "max_z_score": incident.max_z_score,
        "threshold": incident.threshold,
        "message": message,
        "evidence": incident.evidence or {},
        "acknowledged": bool(incident.acknowledged),
        "acknowledged_at": utc_isoformat(incident.acknowledged_at),
    }


def incident_to_alert_dict(incident: Incident) -> Dict:
    data = incident_to_dict(incident)
    return {
        "id": incident.id,
        "incident_id": incident.id,
        "created_at": data["created_at"],
        "stage": incident.primary_stage,
        "severity": incident.severity,
        "anomaly_score": incident.anomaly_score,
        "max_z_score": incident.max_z_score,
        "threshold": incident.threshold,
        "message": data["message"],
        "acknowledged": bool(incident.acknowledged),
        "affected_stages": incident.affected_stages or [],
        "first_detected_stage": incident.first_detected_stage,
        "primary_stage": incident.primary_stage,
        "status": incident.status,
    }


class IncidentService:
    def close_expired(self, db: Session, now: Optional[datetime] = None) -> None:
        now = now or datetime.now(timezone.utc)
        cutoff = now - INCIDENT_COOLDOWN
        open_incidents = db.query(Incident).filter(Incident.status == INCIDENT_STATUS_OPEN).all()
        for incident in open_incidents:
            updated_at = incident.updated_at
            if updated_at and updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            if updated_at and updated_at < cutoff:
                incident.status = INCIDENT_STATUS_CLOSED
                incident.end_time = incident.updated_at

    def process(self, detection: Dict, db: Session) -> Optional[Dict]:
        if not detection.get("is_episode_start"):
            return None

        now = _parse_timestamp(detection.get("timestamp"))
        self.close_expired(db, now)

        incident = self._find_active_incident(db, now)
        created = incident is None
        if created:
            incident = self._create_incident(detection, now)
            db.add(incident)
            db.flush()
            self._create_compatible_alert(incident, db)
        else:
            self._update_incident(incident, detection, now)

        db.flush()
        return {
            "incident": incident_to_dict(incident),
            "alert": incident_to_alert_dict(incident),
            "created": created,
        }

    def _find_active_incident(self, db: Session, now: datetime) -> Optional[Incident]:
        cutoff = now - INCIDENT_COOLDOWN
        return (
            db.query(Incident)
            .filter(Incident.status == INCIDENT_STATUS_OPEN)
            .filter(Incident.updated_at >= cutoff)
            .order_by(Incident.updated_at.desc())
            .first()
        )

    def _create_incident(self, detection: Dict, now: datetime) -> Incident:
        stage = detection["stage"]
        evidence = {
            "stages": {stage: _stage_evidence(detection)},
            "timeline": [{
                "timestamp": now.isoformat(),
                "stage": stage,
                "max_z_score": detection["max_z_score"],
                "threshold": detection["threshold"],
                "anomaly_score": detection["anomaly_score"],
            }],
        }
        return Incident(
            created_at=now,
            updated_at=now,
            start_time=now,
            status=INCIDENT_STATUS_OPEN,
            severity=ALERT_TYPE_DANGER,
            first_detected_stage=stage,
            primary_stage=stage,
            affected_stages=[stage],
            max_z_score=detection["max_z_score"],
            threshold=detection["threshold"],
            anomaly_score=detection["anomaly_score"],
            evidence=evidence,
            acknowledged=False,
        )

    def _update_incident(self, incident: Incident, detection: Dict, now: datetime) -> None:
        stage = detection["stage"]
        affected = list(dict.fromkeys([*(incident.affected_stages or []), stage]))
        evidence = incident.evidence or {"stages": {}, "timeline": []}
        stage_data = evidence.get("stages", {}).get(stage) or _stage_evidence(detection)
        stage_data["last_seen"] = now.isoformat()
        stage_data["max_z_score"] = max(stage_data.get("max_z_score", 0), detection["max_z_score"])
        stage_data["threshold"] = detection["threshold"]
        stage_data["anomaly_score"] = detection["anomaly_score"]
        stage_data["sensor_values"] = detection.get("sensor_values", {})
        stage_data["actuator_values"] = detection.get("actuator_values", {})
        stage_data["raw_values"] = detection.get("raw_values", {})
        evidence.setdefault("stages", {})[stage] = stage_data
        evidence.setdefault("timeline", []).append({
            "timestamp": now.isoformat(),
            "stage": stage,
            "max_z_score": detection["max_z_score"],
            "threshold": detection["threshold"],
            "anomaly_score": detection["anomaly_score"],
        })

        incident.updated_at = now
        incident.affected_stages = affected
        incident.evidence = evidence

        if detection["max_z_score"] > (incident.max_z_score or 0):
            incident.primary_stage = stage
            incident.max_z_score = detection["max_z_score"]
            incident.threshold = detection["threshold"]
            incident.anomaly_score = detection["anomaly_score"]

    def _create_compatible_alert(self, incident: Incident, db: Session) -> None:
        alert = Alert(
            created_at=incident.created_at,
            stage=incident.primary_stage,
            severity=incident.severity,
            anomaly_score=incident.anomaly_score,
            max_z_score=incident.max_z_score,
            threshold=incident.threshold,
            message=_build_message(incident),
            acknowledged=False,
            incident_id=incident.id,
        )
        db.add(alert)


incident_service = IncidentService()
