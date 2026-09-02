"""
===============================================================================
Module: src/ml/predict.py
Project: Low-Cost Real-Time Mine Subsidence Monitoring & Early Warning System
===============================================================================

EDUCATIONAL OVERVIEW:
--------------------
This module provides a production-ready inference interface for the trained
sensor-fusion risk classification model.

Key Inference Workflow:
1. Input Ingestion: Accepts raw sensor dictionaries (from ESP32 HTTP/LoRa packets)
   or Pandas DataFrames.
2. Feature Engineering on the Fly: Automatically derives the physics-based
   sensor features (Resultant Acceleration, Scaled Distance, Kinetic Energy Proxy)
   using `engineer_sensor_features()` from `src.features.vibration_features`.
3. Preprocessing & Scaling: Passes data through the fitted `ColumnTransformer`
   embedded inside the saved Scikit-learn Pipeline (ensuring zero train-test skew).
4. Real Model Probabilities: Extracts the true softmax/softprob confidence distribution
   from the model (`predict_proba`) rather than hardcoding or guessing confidence scores.
5. Standardized Output: Formats the response for downstream alert sirens, SMS gateways,
   and monitoring dashboards.

Inputs:
- Dictionary or DataFrame of incoming sensor measurements.

Outputs:
- Dictionary with predicted `risk_level`, real `confidence`, and class `probabilities`.
===============================================================================
"""

import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import joblib
from typing import Dict, Any, Union
import pandas as pd
import numpy as np

from src.features.vibration_features import engineer_sensor_features

DEFAULT_MODEL_PATH = "models/risk_classifier.joblib"


class RiskPredictor:
    """
    Inference engine for real-time mine vibration and subsidence risk classification.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path.resolve()}. "
                "Please run `python -m src.ml.evaluate` or `python run_pipeline.py` first."
            )

        print(f"[INFERENCE] Loading trained model bundle from: {self.model_path.name}")
        bundle = joblib.load(self.model_path)
        
        self.pipeline = bundle["model_pipeline"]
        self.label_encoder = bundle["label_encoder"]
        self.model_name = bundle["model_name"]
        self.classes = bundle["classes"]
        self.feature_names = bundle["feature_names"]
        print(f"[INFERENCE] Model '{self.model_name}' loaded successfully. Classes: {self.classes}")

    def predict(self, sensor_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Executes real-time inference on sensor readings.

        Args:
            sensor_data (Union[Dict[str, Any], pd.DataFrame]): Input sensor payload.

        Returns:
            Dict[str, Any]: Prediction result containing risk category, confidence, and probabilities.
        """
        # 1. Convert dictionary payload to single-row DataFrame if necessary
        if isinstance(sensor_data, dict):
            df_input = pd.DataFrame([sensor_data])
        else:
            df_input = sensor_data.copy()

        # 2. Apply feature engineering to derive hardware physics metrics
        df_engineered = engineer_sensor_features(df_input)

        # 3. Ensure all expected model feature columns exist (fill missing with 0.0)
        for col in self.feature_names:
            if col not in df_engineered.columns:
                df_engineered[col] = 0.0

        X_inference = df_engineered[self.feature_names]

        # 4. Generate Model Predictions and Probability Distribution
        probabilities = self.pipeline.predict_proba(X_inference)[0]
        predicted_idx = int(np.argmax(probabilities))
        predicted_label = str(self.label_encoder.inverse_transform([predicted_idx])[0])
        confidence = float(probabilities[predicted_idx])

        # Map class probabilities to human-readable dictionary
        prob_dict = {
            str(cls_name): float(round(prob, 4))
            for cls_name, prob in zip(self.classes, probabilities)
        }

        response = {
            "risk_level": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "model_used": self.model_name
        }

        # Include metadata if present
        if "node_id" in df_input.columns and pd.notna(df_input["node_id"].iloc[0]) and df_input["node_id"].iloc[0] not in [None, "None"]:
            response["node_id"] = str(df_input["node_id"].iloc[0])
        else:
            response["node_id"] = None
        if "timestamp" in df_input.columns and pd.notna(df_input["timestamp"].iloc[0]) and df_input["timestamp"].iloc[0] not in [None, "None"]:
            response["timestamp"] = str(df_input["timestamp"].iloc[0])

        return response


# Global singleton predictor instance
_PREDICTOR_INSTANCE = None


def get_predictor(model_path: str = DEFAULT_MODEL_PATH) -> RiskPredictor:
    """
    Returns a cached RiskPredictor instance.
    """
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = RiskPredictor(model_path=model_path)
    return _PREDICTOR_INSTANCE


def predict_risk(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for external callers and API endpoints.
    """
    predictor = get_predictor()
    return predictor.predict(sensor_data)


if __name__ == "__main__":
    from src.ml.evaluate import compare_and_select_best_model
    
    # Ensure model is trained
    if not Path(DEFAULT_MODEL_PATH).exists():
        compare_and_select_best_model()

    # Sample test payload simulating an ESP32 LoRa sensor packet
    sample_esp32_packet = {
        "node_id": "ESP32_NODE_01",
        "acc_x_ms2": 0.31,
        "acc_y_ms2": 0.28,
        "acc_z_ms2": 0.65,
        "temperature_c": 32.5,
        "ppv_mms": 14.8,
        "frequency_hz": 35.2,
        "psd_value": 0.045,
        "geophone_mms": 1.85,
        "seismometer_ms2": 0.42
    }

    print("\n[TEST] Submitting sample hardware sensor payload for inference:")
    result = predict_risk(sample_esp32_packet)
    print("\n[PREDICTION RESULT]:")
    print(f"  Risk Level:    {result['risk_level']}")
    print(f"  Confidence:    {result['confidence'] * 100:.2f}%")
    print(f"  Probabilities: {result['probabilities']}")
    print(f"  Model Engine:  {result['model_used']}")
    if "node_id" in result:
        print(f"  Node ID:       {result['node_id']}")
