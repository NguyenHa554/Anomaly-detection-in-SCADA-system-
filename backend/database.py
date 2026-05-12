"""
database.py — SQLAlchemy setup with SQLite.
To switch to PostgreSQL, change DATABASE_URL to:
    "postgresql://user:password@localhost:5432/scada_db"
and install psycopg2: pip install psycopg2-binary
"""

from sqlalchemy import (
    create_engine, Column, Integer, Float, Boolean,
    String, DateTime, JSON, inspect, text
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from datetime import datetime, timezone

DATABASE_URL = "sqlite:///./scada.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class SensorReading(Base):
    __tablename__ = "sensor_data"

    id              = Column(Integer, primary_key=True, index=True)
    timestamp       = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    stage           = Column(String(10), index=True)
    z_score         = Column(Float, nullable=True)
    is_anomaly      = Column(Boolean, default=False)
    sensor_values   = Column(JSON, nullable=True)
    actuator_values = Column(JSON, nullable=True)
    raw_values      = Column(JSON, nullable=True)


class Anomaly(Base):
    __tablename__ = "anomalies"

    id            = Column(Integer, primary_key=True, index=True)
    timestamp     = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    stage         = Column(String(10))
    anomaly_score = Column(Float)
    max_z_score   = Column(Float)
    threshold     = Column(Float)
    severity      = Column(String(20))
    details       = Column(JSON, nullable=True)


class Alert(Base):
    __tablename__ = "alerts"

    id              = Column(Integer, primary_key=True, index=True)
    created_at      = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    stage           = Column(String(10))
    severity        = Column(String(20))
    anomaly_score   = Column(Float)
    max_z_score     = Column(Float)
    threshold       = Column(Float)
    message         = Column(String)
    acknowledged    = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    incident_id     = Column(Integer, nullable=True, index=True)


class Incident(Base):
    __tablename__ = "incidents"

    id                   = Column(Integer, primary_key=True, index=True)
    created_at           = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at           = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    start_time           = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    end_time             = Column(DateTime(timezone=True), nullable=True)
    status               = Column(String(20), default="OPEN", index=True)
    severity             = Column(String(20), default="DANGER")
    first_detected_stage = Column(String(10))
    primary_stage        = Column(String(10))
    affected_stages      = Column(JSON, nullable=True)
    max_z_score          = Column(Float)
    threshold            = Column(Float)
    anomaly_score        = Column(Float)
    evidence             = Column(JSON, nullable=True)
    acknowledged         = Column(Boolean, default=False)
    acknowledged_at      = Column(DateTime(timezone=True), nullable=True)


def create_tables():
    Base.metadata.create_all(bind=engine)
    _ensure_sensor_data_columns()
    _ensure_alert_columns()


def _ensure_sensor_data_columns():
    inspector = inspect(engine)
    existing_columns = {col["name"] for col in inspector.get_columns("sensor_data")}
    required_columns = {
        "sensor_values": "JSON",
        "actuator_values": "JSON",
        "raw_values": "JSON",
    }

    with engine.begin() as connection:
        for column_name, column_type in required_columns.items():
            if column_name in existing_columns:
                continue
            connection.execute(
                text(f"ALTER TABLE sensor_data ADD COLUMN {column_name} {column_type}")
            )


def _ensure_alert_columns():
    inspector = inspect(engine)
    existing_columns = {col["name"] for col in inspector.get_columns("alerts")}

    with engine.begin() as connection:
        if "incident_id" not in existing_columns:
            connection.execute(text("ALTER TABLE alerts ADD COLUMN incident_id INTEGER"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def clear_runtime_data():
    with SessionLocal() as db:
        db.query(Alert).delete()
        db.query(Anomaly).delete()
        db.query(SensorReading).delete()
        db.query(Incident).delete()
        db.commit()
