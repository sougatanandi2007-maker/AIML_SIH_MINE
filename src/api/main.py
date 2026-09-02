"""
===============================================================================
Module: src/api/main.py
Project: Low-Cost Real-Time Mine Subsidence Monitoring & Early Warning System
===============================================================================

EDUCATIONAL OVERVIEW:
--------------------
This module implements the hardware-aligned REST API using FastAPI and Uvicorn.
It receives telemetry packets transmitted by ESP32 microcontrollers across LoRa gateways.

Key Architecture Principles:
1. HARDWARE-ALIGNED PAYLOAD:
   Accepts ONLY physical sensor telemetry (3-axis acceleration, PPV, frequency,
   PSD, geophone velocity, seismometer, and temperature).
   Does NOT require or expose blast operational parameters (charge weight, burden, spacing, delay).
2. FORWARD-COMPATIBLE HARDWARE EXTENSIONS:
   Accepts optional fields for planned hardware additions (`tilt_x_deg`, `tilt_y_deg`,
   `displacement_mm`, `crack_width_mm`, `node_id`, `timestamp`) without breaking inference.
3. REAL MODEL PROBABILITIES:
   Computes actual softmax/softprob distributions from the trained model pipeline.

Endpoints:
- GET  `/`        : Service metadata & documentation links
- GET  `/health`  : Service readiness and model status check
- POST `/predict` : Real-time risk classification for incoming sensor telemetry
===============================================================================
"""

import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime, timezone
from typing import Dict, Optional, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn

from src.ml.predict import get_predictor, DEFAULT_MODEL_PATH

# Initialize FastAPI Application
app = FastAPI(
    title="Mine Subsidence & Ground Vibration Early Warning API (Hardware-Aligned)",
    description=(
        "Production-ready backend ML inference service for ESP32 + LoRa mine sensor networks. "
        "Predicts ground risk categories (NORMAL, WARNING, CRITICAL) using only physical sensor measurements."
    ),
    version="2.0.0"
)


# -----------------------------------------------------------------------------
# PYDANTIC DATA SCHEMAS (HARDWARE-ALIGNED)
# -----------------------------------------------------------------------------

class SensorReadingPayload(BaseModel):
    """
    Hardware-level sensor telemetry payload matching the 5 public inputs:
    vibration, tilt, temperature, moisture, displacement.
    """
    vibration: float = Field(..., description="Ground dynamic vibration level / acceleration (m/s²)")
    tilt: float = Field(..., description="Inclinometer tilt angle (degrees)")
    temperature: float = Field(..., description="Ambient / rock surface temperature (°C)")
    moisture: float = Field(..., description="Soil moisture level / percentage (%)")
    displacement: float = Field(..., description="Ground displacement (mm)")

    # Optional telemetry metadata
    node_id: Optional[str] = Field(None, description="Unique identifier of transmitting sensor node")
    timestamp: Optional[str] = Field(None, description="ISO-8601 observation timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "vibration": 0.05,
                "tilt": 0.08,
                "temperature": 27.0,
                "moisture": 20.0,
                "displacement": 0.2,
                "node_id": "ESP32_NODE_01",
                "timestamp": "2026-09-03T00:30:00Z"
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    Standardized inference response schema returned to LoRa gateway and monitoring dashboards.
    """
    risk_level: str = Field(..., description="Predicted risk category: NORMAL, WARNING, or CRITICAL")
    confidence: float = Field(..., description="True model probability confidence score for the predicted class [0.0 - 1.0]")
    probabilities: Dict[str, float] = Field(..., description="Complete calibrated class probability distribution")
    model_used: str = Field(..., description="Name of the underlying trained machine learning model")
    node_id: Optional[str] = Field(None, description="Node identifier echoed from request")
    timestamp: Optional[str] = Field(None, description="Timestamp of inference")


class HealthResponse(BaseModel):
    """
    System status response schema.
    """
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    features_count: Optional[int] = None
    architecture: str = "Hardware-Aligned (ESP32/LoRa Compatible)"


# -----------------------------------------------------------------------------
# API ROUTE HANDLERS
# -----------------------------------------------------------------------------

@app.get("/", tags=["Info"])
def root_info():
    """
    Root endpoint displaying service metadata.
    """
    return {
        "project": "Low-Cost Real-Time Mine Subsidence Monitoring & Early Warning System",
        "version": "2.0.0 (Hardware-Aligned)",
        "docs_url": "/docs",
        "health_check": "/health",
        "supported_inputs": "Pure physical sensor telemetry (no blast parameters required)"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint verifying API and model readiness.
    """
    try:
        predictor = get_predictor()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=predictor.model_name,
            features_count=len(predictor.feature_names),
            architecture="Hardware-Aligned (ESP32/LoRa Compatible)"
        )
    except Exception as e:
        return HealthResponse(
            status=f"degraded: {str(e)}",
            model_loaded=False,
            model_name=None,
            features_count=None,
            architecture="Degraded"
        )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK, tags=["Inference"])
def predict_sensor_risk(payload: SensorReadingPayload):
    """
    Predicts ground vibration and mine risk category from physical sensor telemetry.
    Compatible directly with ESP32 edge nodes and LoRa gateways.
    """
    try:
        print("\n" + "=" * 80)
        print(">>> [DEBUG LOG] INCOMING HTTP POST /predict REQUEST <<<")
        print("REQUEST DATA (parsed by Pydantic):", payload.model_dump())
        print("=" * 80)
        predictor = get_predictor()
        sensor_dict = payload.model_dump()
        result = predictor.predict(sensor_dict)

        # Inject timestamp if not present
        if "timestamp" not in result or result["timestamp"] is None or result["timestamp"] == "None":
            result["timestamp"] = datetime.now(timezone.utc).isoformat()

        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
