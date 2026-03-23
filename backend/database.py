"""
database.py — SQLAlchemy setup with SQLite.
To switch to PostgreSQL, change DATABASE_URL to:
    "postgresql://user:password@localhost:5432/scada_db"
and install psycopg2: pip install psycopg2-binary
"""

from sqlalchemy import (
    create_engine, Column, Integer, Float, Boolean,
    String, DateTime, JSON
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

    id        = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    stage     = Column(String(10), index=True)
    z_score   = Column(Float, nullable=True)
    is_anomaly = Column(Boolean, default=False)


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


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
