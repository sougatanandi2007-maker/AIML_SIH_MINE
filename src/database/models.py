"""
===============================================================================
Module: src/database/models.py
Project: Low-Cost Real-Time Mine Subsidence Monitoring & Early Warning System
===============================================================================

EDUCATIONAL OVERVIEW:
--------------------
In production IoT and mining sensor deployments, incoming sensor telemetry must be
persisted reliably in a relational database for auditing, historical time-series
analysis, and model retraining.

We use SQLAlchemy Object Relational Mapping (ORM) to define our database schema
in pure Python classes. Each instance of `SensorObservation` represents one row
in the `sensor_observations` table.

Why SQLite for the Prototype?
- Serverless, zero-configuration, single-file storage (`mine_monitoring.db`).
- Seamless migration to PostgreSQL / TimescaleDB in production with zero ORM code changes.

Fields mapped directly from our discovered Kaggle dataset:
- id: Primary Key (auto-incrementing unique identifier)
- timestamp: Observation datetime
- blast_id: Identifier for mining blast event
- charge_weight_kg: Explosive charge weight in kilograms
- burden_m: Distance from blasthole to free rock face in meters
- spacing_m: Distance between adjacent blastholes in meters
- delay_ms: Millisecond detonation delay
- soil_type: Geological soil/rock classification ('Hard', 'Medium', 'Soft')
- temperature_c: Ambient ambient temperature in Celsius
- wind_speed_ms: Environmental wind velocity in m/s
- seismometer_ms2: Ground seismic acceleration
- geophone_mms: Ground particle velocity from geophone in mm/s
- acc_x_ms2, acc_y_ms2, acc_z_ms2: Triaxial accelerometer values in m/s²
- psd_value: Power Spectral Density energy metric
- ppv_mms: Peak Particle Velocity in mm/s
- frequency_hz: Dominant oscillation frequency in Hertz
- vibration_level: Original dataset vibration category ('Low', 'Medium', 'High')
- risk_label: Prototype standardized risk category ('NORMAL', 'WARNING', 'CRITICAL')
- created_at: System record creation timestamp
===============================================================================
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class SensorObservation(Base):
    """
    SQLAlchemy ORM Model representing an individual sensor telemetry record.
    """
    __tablename__ = "sensor_observations"

    id = Column(Integer, primary_key=True, autoincrement=True, doc="Unique row identifier")
    timestamp = Column(DateTime, nullable=True, doc="Recorded sensor timestamp")
    blast_id = Column(String(50), nullable=True, doc="Associated blast identifier")
    charge_weight_kg = Column(Float, nullable=True, doc="Explosive charge mass (kg)")
    burden_m = Column(Float, nullable=True, doc="Burden distance (meters)")
    spacing_m = Column(Float, nullable=True, doc="Hole spacing (meters)")
    delay_ms = Column(Float, nullable=True, doc="Blast delay (ms)")
    soil_type = Column(String(50), nullable=True, doc="Geological soil/strata type")
    temperature_c = Column(Float, nullable=True, doc="Ambient temperature (°C)")
    wind_speed_ms = Column(Float, nullable=True, doc="Wind speed (m/s)")
    seismometer_ms2 = Column(Float, nullable=True, doc="Seismometer ground acceleration")
    geophone_mms = Column(Float, nullable=True, doc="Geophone particle velocity (mm/s)")
    acc_x_ms2 = Column(Float, nullable=True, doc="X-axis acceleration (m/s²)")
    acc_y_ms2 = Column(Float, nullable=True, doc="Y-axis acceleration (m/s²)")
    acc_z_ms2 = Column(Float, nullable=True, doc="Z-axis acceleration (m/s²)")
    psd_value = Column(Float, nullable=True, doc="Power spectral density value")
    ppv_mms = Column(Float, nullable=True, doc="Peak Particle Velocity (mm/s)")
    frequency_hz = Column(Float, nullable=True, doc="Dominant frequency (Hz)")
    vibration_level = Column(String(50), nullable=True, doc="Original vibration label")
    risk_label = Column(String(50), nullable=True, doc="Normalized risk category (NORMAL, WARNING, CRITICAL)")
    created_at = Column(DateTime, default=datetime.utcnow, doc="System ingestion timestamp")

    def __repr__(self) -> str:
        return (
            f"<SensorObservation(id={self.id}, blast_id='{self.blast_id}', "
            f"ppv={self.ppv_mms}, risk_label='{self.risk_label}')>"
        )
