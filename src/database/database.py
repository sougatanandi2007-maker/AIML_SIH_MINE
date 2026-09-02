"""
===============================================================================
Module: src/database/database.py
Project: Low-Cost Real-Time Mine Subsidence Monitoring & Early Warning System
===============================================================================

EDUCATIONAL OVERVIEW:
--------------------
This module manages database connections, table creation, batch insertions,
and data retrieval.

Key Components:
1. `create_engine`: Sets up the low-level connection pool to SQLite (`mine_monitoring.db`).
2. `sessionmaker`: Generates isolated transactional database sessions.
3. `init_db()`: Creates the physical SQLite database and tables based on the SQLAlchemy models.
4. `insert_sensor_data()`: Batch-inserts a Pandas DataFrame into the SQLite database.
5. `load_sensor_data_to_df()`: Executes a SQL query to load stored telemetry records back
   into a clean Pandas DataFrame for training or analytics.

Inputs:
- Database URL (default: 'sqlite:///mine_monitoring.db')
- Pandas DataFrames containing sensor records.

Outputs:
- SQLite database file `mine_monitoring.db`
- Ingested row count and retrieved DataFrames.
===============================================================================
"""

import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

from src.database.models import Base, SensorObservation

load_dotenv()

DEFAULT_DB_URL = os.getenv("DATABASE_URL", "sqlite:///mine_monitoring.db")


def get_engine(db_url: Optional[str] = None):
    """
    Creates and returns a SQLAlchemy engine.
    """
    url = db_url or DEFAULT_DB_URL
    return create_engine(url, echo=False)


def get_session_factory(engine=None):
    """
    Returns a configured sessionmaker bound to the given engine.
    """
    eng = engine or get_engine()
    return sessionmaker(bind=eng)


def init_db(db_url: Optional[str] = None):
    """
    Initializes database tables defined in SQLAlchemy ORM models.
    """
    engine = get_engine(db_url)
    print(f"[DATABASE] Initializing database schema at: {engine.url}")
    Base.metadata.create_all(bind=engine)
    print("[DATABASE] Tables created successfully.")


def insert_sensor_data(df: pd.DataFrame, db_url: Optional[str] = None) -> int:
    """
    Batch-inserts sensor telemetry DataFrame into SQLite database.

    Args:
        df (pd.DataFrame): Processed sensor DataFrame.
        db_url (Optional[str]): Database connection string.

    Returns:
        int: Number of rows inserted.
    """
    engine = get_engine(db_url)
    session_factory = get_session_factory(engine)
    
    # Ensure tables exist
    init_db(db_url)

    # Map DataFrame columns to ORM fields
    records = []
    for _, row in df.iterrows():
        # Handle timestamp conversion
        ts = None
        if "timestamp" in row and pd.notnull(row["timestamp"]):
            try:
                ts = pd.to_datetime(row["timestamp"])
            except Exception:
                ts = None

        record = SensorObservation(
            timestamp=ts,
            blast_id=str(row.get("blast_id", "")),
            charge_weight_kg=float(row.get("charge_weight_kg", 0.0)),
            burden_m=float(row.get("burden_m", 0.0)),
            spacing_m=float(row.get("spacing_m", 0.0)),
            delay_ms=float(row.get("delay_ms", 0.0)),
            soil_type=str(row.get("soil_type", "")),
            temperature_c=float(row.get("temperature_c", 0.0)),
            wind_speed_ms=float(row.get("wind_speed_ms", 0.0)),
            seismometer_ms2=float(row.get("seismometer_ms2", 0.0)),
            geophone_mms=float(row.get("geophone_mms", 0.0)),
            acc_x_ms2=float(row.get("acc_x_ms2", 0.0)),
            acc_y_ms2=float(row.get("acc_y_ms2", 0.0)),
            acc_z_ms2=float(row.get("acc_z_ms2", 0.0)),
            psd_value=float(row.get("psd_value", 0.0)),
            ppv_mms=float(row.get("ppv_mms", 0.0)),
            frequency_hz=float(row.get("frequency_hz", 0.0)),
            vibration_level=str(row.get("vibration_level", "")),
            risk_label=str(row.get("risk_label", ""))
        )
        records.append(record)

    session: Session = session_factory()
    try:
        # Clear existing observations if re-populating prototype
        session.query(SensorObservation).delete()
        session.bulk_save_objects(records)
        session.commit()
        inserted_count = len(records)
        print(f"[DATABASE] Successfully inserted {inserted_count:,} sensor observation records.")
        return inserted_count
    except Exception as e:
        session.rollback()
        print(f"[ERROR] Database insertion failed: {e}")
        raise e
    finally:
        session.close()


def load_sensor_data_to_df(db_url: Optional[str] = None) -> pd.DataFrame:
    """
    Reads stored sensor observation records from SQLite database into a Pandas DataFrame.

    Args:
        db_url (Optional[str]): Database connection string.

    Returns:
        pd.DataFrame: Retrieved sensor records.
    """
    engine = get_engine(db_url)
    query = "SELECT * FROM sensor_observations"
    df = pd.read_sql(query, con=engine)
    print(f"[DATABASE] Loaded {len(df):,} records from database.")
    return df


if __name__ == "__main__":
    init_db()
    print("Database connection test complete.")
