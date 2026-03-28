"""
main.py — FastAPI application entry point.
Start with: uvicorn backend.main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from backend.database import create_tables, get_db, Alert
from backend.services.anomaly_detector import detector, STAGE_FEATURES
from backend.services.data_pipeline import process_row, reset_runtime_state


# ── WebSocket Manager ─────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    try:
        detector.load()
        print("Models loaded successfully.")
    except FileNotFoundError:
        print(
            "WARNING: Model files not found in ml_models/. "
            "Run `python export_models.py` first."
        )
    yield


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="SCADA Anomaly Detection API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class SensorRow(BaseModel):
    model_config = {"extra": "allow"}   # accept any sensor columns


class AlertOut(BaseModel):
    id: int
    created_at: datetime
    stage: str
    severity: str
    anomaly_score: float
    max_z_score: float
    threshold: float
    message: str
    acknowledged: bool

    class Config:
        from_attributes = True


# ── REST endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/status")
def get_status():
    stages = {}
    for stage in STAGE_FEATURES:
        buf = detector.buffers.get(stage, [])
        stages[stage] = {
            "buffer_fill":   len(buf),
            "buffer_needed": detector.time_steps + 1,
            "ready":         len(buf) >= detector.time_steps + 1,
        }
    return {
        "model_loaded": detector.is_loaded,
        "loaded_stages": list(detector.loaded_stages),
        "stages":       stages,
        "server_time":  datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/ingest")
async def ingest(row: SensorRow, db: Session = Depends(get_db)):
    if not detector.is_loaded:
        raise HTTPException(503, "Models not loaded yet. Run export_models.py first.")
    result = await process_row(row.model_dump(), db, manager.broadcast)
    return result


@app.post("/api/runtime/reload")
def reload_runtime():
    create_tables()
    reset_runtime_state()
    detector.load()
    return {
        "ok": True,
        "loaded_stages": list(detector.loaded_stages),
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/alerts", response_model=List[AlertOut])
def get_alerts(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    return (
        db.query(Alert)
        .order_by(Alert.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@app.post("/api/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(404, "Alert not found.")
    alert.acknowledged    = True
    alert.acknowledged_at = datetime.now(timezone.utc)
    db.commit()
    return {"ok": True, "id": alert_id}


@app.get("/api/history")
def get_history(stage: str = None, limit: int = 500, db: Session = Depends(get_db)):
    from backend.database import SensorReading
    q = db.query(SensorReading).order_by(SensorReading.timestamp.desc())
    if stage:
        q = q.filter(SensorReading.stage == stage)
    rows = q.limit(limit).all()
    return [
        {
            "id":             r.id,
            "timestamp":      r.timestamp.isoformat() if r.timestamp else None,
            "stage":          r.stage,
            "z_score":        r.z_score,
            "is_anomaly": r.is_anomaly,
            "sensor_values":  r.sensor_values,
            "actuator_values": r.actuator_values,
            "raw_values":     r.raw_values,
        }
        for r in reversed(rows)
    ]


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()   # keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ── Serve React build (production) ────────────────────────────────────────────
FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_react(full_path: str):
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
