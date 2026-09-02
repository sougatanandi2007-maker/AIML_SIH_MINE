"""
===============================================================================
Module: tests/test_pipeline.py
Project: Low-Cost Real-Time Mine Subsidence Monitoring & Early Warning System
===============================================================================

EDUCATIONAL OVERVIEW:
--------------------
Automated test suite verifying the hardware-aligned architecture:
1. Test 1: Vibration & Waveform Math (RMS, P2P, Crest Factor)
2. Test 2: Hardware Feature Engineering (3D Resultant Acceleration, Kinetic Energy Proxy, Wave Ratio)
3. Test 3: Relational Database Storage & Query Roundtrip (SQLAlchemy SQLite)
4. Test 4: Hardware-Aligned ML Prediction & Probability Distributions
5. Test 5: FastAPI Endpoints (GET /health and POST /predict with hardware payload)
===============================================================================
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.features.vibration_features import (
    extract_waveform_features,
    engineer_hardware_features,
    FINAL_HARDWARE_FEATURE_NAMES
)
from src.database.database import init_db, insert_sensor_data, load_sensor_data_to_df
from src.ml.predict import get_predictor, predict_risk
from src.api.main import app


def test_waveform_math():
    """
    Validates physical feature extraction formulas on synthetic signals.
    """
    t = np.linspace(0, 1, 100)
    signal = 2.0 * np.sin(2 * np.pi * 5 * t)

    metrics = extract_waveform_features(signal)

    # Pure sine wave: peak = 2.0, RMS = 2.0 / sqrt(2) approx 1.414
    assert np.isclose(metrics["peak_amplitude"], 2.0, atol=0.05)
    assert np.isclose(metrics["rms"], 2.0 / np.sqrt(2), atol=0.05)
    assert np.isclose(metrics["peak_to_peak"], 4.0, atol=0.1)
    assert "crest_factor" in metrics
    assert "signal_energy" in metrics


def test_feature_engineering_tabular():
    """
    Validates 3D resultant acceleration, kinetic energy proxy, and wave ratio.
    """
    df = pd.DataFrame([{
        "acc_x_ms2": 3.0,
        "acc_y_ms2": 4.0,
        "acc_z_ms2": 0.0,
        "ppv_mms": 10.0,
        "seismometer_ms2": 2.0,
        "geophone_mms": 1.0,
        "psd_value": 0.5,
        "frequency_hz": 20.0,
        "temperature_c": 28.0
    }])

    df_feat = engineer_hardware_features(df)

    # 3-4-5 triangle for 3D acceleration: sqrt(3^2 + 4^2 + 0^2) = 5.0
    assert np.isclose(df_feat["vibration_magnitude_ms2"].iloc[0], 5.0, atol=1e-3)
    # Horizontal shear acceleration: sqrt(3^2 + 4^2) = 5.0
    assert np.isclose(df_feat["vibration_horizontal_ms2"].iloc[0], 5.0, atol=1e-3)
    # Kinetic energy proxy: 0.5 * 10^2 = 50.0
    assert np.isclose(df_feat["kinetic_energy_proxy"].iloc[0], 50.0, atol=1e-3)
    # Wave ratio: 2.0 / (1.0 + 1e-4) approx 2.0
    assert np.isclose(df_feat["accel_to_velocity_ratio"].iloc[0], 2.0, atol=1e-2)
    # Spectral power product: 0.5 * 20.0 = 10.0
    assert np.isclose(df_feat["spectral_power_product"].iloc[0], 10.0, atol=1e-3)


def test_database_roundtrip(tmp_path):
    """
    Tests SQLite initialization, insertion, and querying.
    """
    test_db_file = tmp_path / "test_mine.db"
    db_url = f"sqlite:///{test_db_file.resolve()}"

    init_db(db_url)

    sample_df = pd.DataFrame([{
        "timestamp": "2026-09-03 08:00:00",
        "blast_id": "TEST_NODE_01",
        "soil_type": "Hard",
        "temperature_c": 25.0,
        "seismometer_ms2": 0.1,
        "geophone_mms": 0.8,
        "acc_x_ms2": 0.1,
        "acc_y_ms2": 0.1,
        "acc_z_ms2": 0.2,
        "psd_value": 0.01,
        "ppv_mms": 1.5,
        "frequency_hz": 25.0,
        "vibration_level": "Low",
        "risk_label": "NORMAL"
    }])

    inserted = insert_sensor_data(sample_df, db_url=db_url)
    assert inserted == 1

    retrieved_df = load_sensor_data_to_df(db_url=db_url)
    assert len(retrieved_df) == 1
    assert retrieved_df["risk_label"].iloc[0] == "NORMAL"


def test_model_prediction():
    """
    Tests hardware-aligned model inference returns valid risk categories and calibrated probabilities.
    """
    predictor = get_predictor()

    # Pure hardware telemetry packet from ESP32
    sample_payload = {
        "node_id": "ESP32_NODE_01",
        "acc_x_ms2": 0.04,
        "acc_y_ms2": 0.03,
        "acc_z_ms2": 0.05,
        "temperature_c": 24.5,
        "ppv_mms": 1.45,
        "frequency_hz": 22.0,
        "psd_value": 0.22,
        "geophone_mms": 1.40,
        "seismometer_ms2": 0.92
    }

    result = predictor.predict(sample_payload)

    assert result["risk_level"] in ["NORMAL", "WARNING", "CRITICAL"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["probabilities"]) == 3
    # Probabilities must sum to ~1.0
    total_prob = sum(result["probabilities"].values())
    assert np.isclose(total_prob, 1.0, atol=0.05)
    assert result["node_id"] == "ESP32_NODE_01"


def test_fastapi_endpoints():
    """
    Tests FastAPI /health and /predict endpoints using TestClient.
    """
    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    health_data = res_health.json()
    assert health_data["status"] == "ok"
    assert health_data["model_loaded"] is True
    assert health_data["features_count"] == len(FINAL_HARDWARE_FEATURE_NAMES)

    # 2. Predict endpoint with 5-field physical sensor readings
    payload = {
        "vibration": 1.2,
        "tilt": 4.0,
        "temperature": 35.0,
        "moisture": 85.0,
        "displacement": 10.0,
        "node_id": "ESP32_TEST_GATEWAY"
    }

    res_predict = client.post("/predict", json=payload)
    assert res_predict.status_code == 200
    predict_data = res_predict.json()
    assert predict_data["risk_level"] in ["NORMAL", "WARNING", "CRITICAL"]
    assert "confidence" in predict_data
    assert "probabilities" in predict_data
    assert predict_data["node_id"] == "ESP32_TEST_GATEWAY"
    assert predict_data["model_used"] in ["Random Forest", "XGBoost"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
