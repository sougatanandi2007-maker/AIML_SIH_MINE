"""
===============================================================================
Module: src/features/vibration_features.py
Project: Low-Cost Real-Time Mine Subsidence Monitoring & Early Warning System
===============================================================================

EDUCATIONAL OVERVIEW:
--------------------
This module handles feature engineering for both:
1. TABULAR HARDWARE SENSOR FUSION:
   Derives physical geomechanical metrics exclusively from sensor telemetry
   measurable by physical edge hardware (triaxial accelerometers, geophones,
   seismometers, and thermal sensors).
   * EXCLUDES all blast operational parameters (charge weight, burden, spacing, delay)
     to prevent train-serving mismatch in real ESP32 deployments.
2. CONTINUOUS WAVEFORM / TIME-SERIES BUFFER PROCESSING:
   Statistical signal processing functions (RMS, Peak-to-Peak, Crest Factor)
   ready for high-frequency continuous ADC sampling on the ESP32.

Hardware Mathematical Formulas & Geotechnical Meaning:
-------------------------------------------------------
1. 3D Resultant Dynamic Acceleration (vibration_magnitude_ms2):
   Formula: a_res = sqrt(a_x^2 + a_y^2 + a_z^2)
   Physical Meaning: Total dynamic shock vector magnitude in 3D space, invariant
   to the physical mounting angle or inclination of the sensor enclosure.

2. Horizontal Dynamic Shear Acceleration (vibration_horizontal_ms2):
   Formula: a_horiz = sqrt(a_x^2 + a_y^2)
   Physical Meaning: Dynamic shear force parallel to the ground surface,
   responsible for lateral slope slippage, bench failure, and pillar shear.

3. Dynamic Ground Kinetic Energy Proxy (kinetic_energy_proxy):
   Formula: E_k = 0.5 * (PPV)^2
   Physical Meaning: Kinetic energy transmitted into the rock mass per unit mass.

4. Dynamic Wave Ratio (accel_to_velocity_ratio):
   Formula: Ratio = |a_seismo| / (|v_geophone| + 1e-4)
   Physical Meaning: Ratio of ground acceleration to particle velocity, indicating
   wavefront steepness and dynamic impedance.

5. Spectral Energy Interaction (spectral_power_product):
   Formula: SPP = PSD * Frequency
   Physical Meaning: Concentrated energy density at the dominant resonance frequency.
===============================================================================
"""

import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, List, Union, Tuple
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1. HARDWARE SENSOR SPECIFICATIONS
# -----------------------------------------------------------------------------

# Primary telemetry channels provided by physical ESP32 edge nodes
HARDWARE_RAW_SENSOR_COLUMNS = [
    "acc_x_ms2",
    "acc_y_ms2",
    "acc_z_ms2",
    "temperature_c",
    "ppv_mms",
    "frequency_hz",
    "psd_value",
    "geophone_mms",
    "seismometer_ms2"
]

# Features derived strictly from the above hardware measurements
HARDWARE_DERIVED_FEATURE_COLUMNS = [
    "vibration_magnitude_ms2",
    "vibration_horizontal_ms2",
    "kinetic_energy_proxy",
    "accel_to_velocity_ratio",
    "spectral_power_product"
]

# Final 14-feature matrix for hardware-aligned model training and inference
FINAL_HARDWARE_FEATURE_NAMES = HARDWARE_RAW_SENSOR_COLUMNS + HARDWARE_DERIVED_FEATURE_COLUMNS


# -----------------------------------------------------------------------------
# 2. CONTINUOUS WAVEFORM SIGNAL PROCESSING (ESP32 ADC BUFFERS)
# -----------------------------------------------------------------------------

def extract_waveform_features(signal_window: Union[np.ndarray, List[float]]) -> Dict[str, float]:
    """
    Computes statistical and physical signal features from a 1D raw waveform buffer.
    Designed for ESP32 high-speed continuous accelerometer / geophone sampling (e.g. 500-1000 Hz).

    Args:
        signal_window (Union[np.ndarray, List[float]]): 1D array of waveform samples.

    Returns:
        Dict[str, float]: Extracted time-domain vibration metrics.
    """
    signal = np.asarray(signal_window, dtype=np.float64)
    if len(signal) == 0:
        return {}

    n = len(signal)
    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))
    min_val = float(np.min(signal))
    max_val = float(np.max(signal))
    peak_amplitude = float(np.max(np.abs(signal)))
    peak_to_peak = float(max_val - min_val)

    # RMS = sqrt( (1/N) * sum(x^2) )
    rms_val = float(np.sqrt(np.mean(signal ** 2)))

    # Signal Energy = sum(x^2)
    signal_energy = float(np.sum(signal ** 2))

    # Crest Factor = Peak / RMS (handles zero division)
    crest_factor = float(peak_amplitude / (rms_val + 1e-8))

    # Zero-Crossing Rate (ZCR): rate of sign-changes along the signal
    zero_crossings = int(np.sum(np.diff(np.sign(signal) != 0)))
    zcr = float(zero_crossings / max(1, n - 1))

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "peak_amplitude": peak_amplitude,
        "peak_to_peak": peak_to_peak,
        "rms": rms_val,
        "signal_energy": signal_energy,
        "crest_factor": crest_factor,
        "zero_crossing_rate": zcr
    }


# -----------------------------------------------------------------------------
# 3. HARDWARE-ALIGNED FEATURE ENGINEERING
# -----------------------------------------------------------------------------

def engineer_hardware_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives physical geomechanical features strictly from hardware-compatible
    measurements. Does NOT use blast or geological metadata.

    Args:
        df (pd.DataFrame): DataFrame containing raw sensor columns or 5-sensor telemetry.

    Returns:
        pd.DataFrame: DataFrame augmented with derived hardware features.
    """
    df_feat = df.copy()

    # Map high-level sensor inputs to appropriate hardware features if present
    if "temperature" in df_feat.columns and "temperature_c" not in df_feat.columns:
        df_feat["temperature_c"] = df_feat["temperature"]

    if "tilt" in df_feat.columns and "tilt_x_deg" not in df_feat.columns:
        df_feat["tilt_x_deg"] = df_feat["tilt"]
        if "tilt_y_deg" not in df_feat.columns:
            df_feat["tilt_y_deg"] = 0.0

    if "displacement" in df_feat.columns and "displacement_mm" not in df_feat.columns:
        df_feat["displacement_mm"] = df_feat["displacement"]

    # Map vibration input into existing vibration feature-engineering logic
    if "vibration" in df_feat.columns:
        tilt_deg = df_feat["tilt_x_deg"] if "tilt_x_deg" in df_feat.columns else (df_feat["tilt"] if "tilt" in df_feat.columns else 0.0)
        tilt_rad = np.radians(tilt_deg)

        if "vibration_magnitude_ms2" not in df_feat.columns:
            df_feat["vibration_magnitude_ms2"] = df_feat["vibration"]

        if not all(col in df_feat.columns for col in ["acc_x_ms2", "acc_y_ms2", "acc_z_ms2"]):
            horiz_shear = np.abs(df_feat["vibration"] * np.sin(tilt_rad))
            df_feat["acc_x_ms2"] = np.maximum(horiz_shear, df_feat["vibration"] / np.sqrt(3))
            df_feat["acc_y_ms2"] = df_feat["vibration"] / np.sqrt(3)
            df_feat["acc_z_ms2"] = np.abs(df_feat["vibration"] * np.cos(tilt_rad))

        if "vibration_horizontal_ms2" not in df_feat.columns:
            df_feat["vibration_horizontal_ms2"] = np.sqrt(df_feat["acc_x_ms2"] ** 2 + df_feat["acc_y_ms2"] ** 2)

        # In ground vibration geophysics, velocity (PPV in mm/s) = acceleration / (2*pi*f)
        # For dominant ground resonance (~25-30 Hz), PPV (mm/s) ≈ 28.0 * vibration (m/s²)
        if "ppv_mms" not in df_feat.columns:
            df_feat["ppv_mms"] = df_feat["vibration"] * 28.0

        if "kinetic_energy_proxy" not in df_feat.columns:
            df_feat["kinetic_energy_proxy"] = 0.5 * (df_feat["ppv_mms"] ** 2)

        if "geophone_mms" not in df_feat.columns:
            df_feat["geophone_mms"] = df_feat["ppv_mms"] * 0.6
        if "seismometer_ms2" not in df_feat.columns:
            df_feat["seismometer_ms2"] = df_feat["vibration"]
        if "frequency_hz" not in df_feat.columns:
            df_feat["frequency_hz"] = 28.0
        if "psd_value" not in df_feat.columns:
            df_feat["psd_value"] = 0.01 * (df_feat["vibration"] ** 2)

    # 1. 3D Resultant Dynamic Acceleration: a_res = sqrt(ax^2 + ay^2 + az^2)
    if all(col in df_feat.columns for col in ["acc_x_ms2", "acc_y_ms2", "acc_z_ms2"]):
        df_feat["vibration_magnitude_ms2"] = np.sqrt(
            df_feat["acc_x_ms2"] ** 2 +
            df_feat["acc_y_ms2"] ** 2 +
            df_feat["acc_z_ms2"] ** 2
        )
        # 2. Horizontal Dynamic Shear Acceleration: a_horiz = sqrt(ax^2 + ay^2)
        df_feat["vibration_horizontal_ms2"] = np.sqrt(
            df_feat["acc_x_ms2"] ** 2 +
            df_feat["acc_y_ms2"] ** 2
        )
    elif "vibration_magnitude_ms2" not in df_feat.columns:
        df_feat["vibration_magnitude_ms2"] = 0.0
        df_feat["vibration_horizontal_ms2"] = 0.0

    # 3. Dynamic Ground Kinetic Energy Proxy: E_k = 0.5 * PPV^2
    if "ppv_mms" in df_feat.columns:
        df_feat["kinetic_energy_proxy"] = 0.5 * (df_feat["ppv_mms"] ** 2)
    elif "kinetic_energy_proxy" not in df_feat.columns:
        df_feat["kinetic_energy_proxy"] = 0.0

    # 4. Dynamic Wave Ratio: Seismometer acceleration to Geophone velocity
    if "seismometer_ms2" in df_feat.columns and "geophone_mms" in df_feat.columns:
        safe_geophone = np.maximum(np.abs(df_feat["geophone_mms"]), 0.0001)
        df_feat["accel_to_velocity_ratio"] = np.abs(df_feat["seismometer_ms2"]) / safe_geophone
    elif "accel_to_velocity_ratio" not in df_feat.columns:
        df_feat["accel_to_velocity_ratio"] = 0.0

    # 5. Spectral Energy Interaction: PSD * Dominant Frequency
    if "psd_value" in df_feat.columns and "frequency_hz" in df_feat.columns:
        df_feat["spectral_power_product"] = df_feat["psd_value"] * df_feat["frequency_hz"]
    elif "spectral_power_product" not in df_feat.columns:
        df_feat["spectral_power_product"] = 0.0

    return df_feat


def prepare_hardware_ml_features(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Selects the 14 hardware-compatible features (X) and target risk label (y).
    Strictly filters out any non-hardware blast parameters.

    Returns:
        Tuple[pd.DataFrame, pd.Series, List[str]]: (X, y, feature_names)
    """
    df_engineered = engineer_hardware_features(df)

    # Ensure all 14 features exist
    for col in FINAL_HARDWARE_FEATURE_NAMES:
        if col not in df_engineered.columns:
            df_engineered[col] = 0.0

    X = df_engineered[FINAL_HARDWARE_FEATURE_NAMES].copy()

    # Extract target label
    if "risk_label" in df_engineered.columns:
        y = df_engineered["risk_label"].copy()
    elif "vibration_level" in df_engineered.columns:
        from src.data.preprocess import RISK_LEVEL_MAPPING
        y = df_engineered["vibration_level"].map(RISK_LEVEL_MAPPING).fillna("WARNING")
    else:
        y = pd.Series(["NORMAL"] * len(df_engineered))

    return X, y, FINAL_HARDWARE_FEATURE_NAMES


# Alias for backwards compatibility
prepare_ml_features = prepare_hardware_ml_features
engineer_sensor_features = engineer_hardware_features


if __name__ == "__main__":
    from src.data.preprocess import preprocess_dataset

    processed_file = Path("data/processed/processed_all.csv")
    if not processed_file.exists():
        preprocess_dataset()

    df_sample = pd.read_csv(processed_file)
    X, y, feat_names = prepare_hardware_ml_features(df_sample)

    print("\n" + "=" * 80)
    print("HARDWARE-ALIGNED FEATURE MATRIX SUMMARY")
    print("=" * 80)
    print(f"Total Hardware Features: {len(feat_names)}")
    print(f"Feature Names: {feat_names}")
    print(f"X Shape: {X.shape}")
    print(f"Target Distribution:\n{y.value_counts()}")
    print("=" * 80)
