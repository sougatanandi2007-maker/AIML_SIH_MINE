"""
===============================================================================
Module: src/data/preprocess.py
Project: Low-Cost Real-Time Mine Subsidence Monitoring & Early Warning System
===============================================================================

EDUCATIONAL OVERVIEW:
--------------------
Data preprocessing converts noisy, unstandardized raw sensor logs into a clean,
structured format suitable for both database storage and machine learning.

Preprocessing Decisions & Scientific Rationale:
-----------------------------------------------
1. Column Standardization:
   Raw headers contain special characters, units, and encoding quirks
   (e.g., 'Temperature(°C)', 'Acc_X(m/s²)'). We normalize all names into
   standard lowercase snake_case (e.g., 'temperature_c', 'acc_x_ms2')
   to avoid syntax issues across SQL engines and serialization frameworks.

2. Missing Value Handling (Imputation):
   - Numeric features: Imputed using the median. The median is robust against
     extreme values and skewness, unlike the mean.
   - Categorical features: Imputed using the statistical mode (most frequent value).
   - Rationale: While our raw dataset may have zero nulls, real IoT sensors
     inevitably experience packet drops. Establishing a pipeline imputation
     strategy ensures production robustness.

3. Duplicate Removal:
   Duplicate records are filtered out to prevent data leakage and avoid
   artificially inflating model validation metrics.

4. Outlier Inspection via Interquartile Range (IQR):
   IQR = Q3 - Q1. Values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are flagged.
   - Rationale: In mining blasts and ground vibration, high PPV and dynamic shock
     are physically authentic event spikes, NOT measurement errors. Therefore,
     we inspect and log outliers but DO NOT blindly delete them.

5. Target Standardization & Mapping:
   Original 'Vibration_Level' ('Low', 'Medium', 'High') is mapped to prototype
   operational risk categories:
   - 'Low'    -> 'NORMAL'   (Safe baseline ground motion)
   - 'Medium' -> 'WARNING'  (Elevated vibration, threshold advisory)
   - 'High'   -> 'CRITICAL' (Severe dynamic shock, immediate alert)
   * Note: This is an empirical prototype risk mapping for vibration monitoring.

6. Stratified Train/Test Split:
   An 80/20 split stratified by the target risk class ensures equal representation
   of minority classes (such as CRITICAL/High risk) in both training and test sets.

Outputs:
- data/processed/processed_all.csv
- data/processed/train.csv
- data/processed/test.csv
- SQLite ingestion in mine_monitoring.db
===============================================================================
"""

import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.database.database import insert_sensor_data


# Mapping from raw column variants to clean standardized snake_case
COLUMN_NAME_MAPPING = {
    "Timestamp": "timestamp",
    "Blast_ID": "blast_id",
    "Charge_Weight(kg)": "charge_weight_kg",
    "Burden(m)": "burden_m",
    "Spacing(m)": "spacing_m",
    "Delay(ms)": "delay_ms",
    "Soil_Type": "soil_type",
    "Temperature(°C)": "temperature_c",
    "Temperature(C)": "temperature_c",
    "Temperature(C)": "temperature_c",
    "Wind_Speed(m/s)": "wind_speed_ms",
    "Seismometer(m/s²)": "seismometer_ms2",
    "Seismometer(m/s)": "seismometer_ms2",
    "Seismometer(m/s)": "seismometer_ms2",
    "Geophone(mm/s)": "geophone_mms",
    "Acc_X(m/s²)": "acc_x_ms2",
    "Acc_X(m/s)": "acc_x_ms2",
    "Acc_X(m/s)": "acc_x_ms2",
    "Acc_Y(m/s²)": "acc_y_ms2",
    "Acc_Y(m/s)": "acc_y_ms2",
    "Acc_Y(m/s)": "acc_y_ms2",
    "Acc_Z(m/s²)": "acc_z_ms2",
    "Acc_Z(m/s)": "acc_z_ms2",
    "Acc_Z(m/s)": "acc_z_ms2",
    "PSD_Value": "psd_value",
    "PPV(mm/s)": "ppv_mms",
    "Frequency(Hz)": "frequency_hz",
    "Vibration_Level": "vibration_level"
}

# Prototype Risk Level Mapping
RISK_LEVEL_MAPPING = {
    "Low": "NORMAL",
    "Medium": "WARNING",
    "High": "CRITICAL"
}


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames raw columns to standard snake_case identifiers.
    """
    new_columns = {}
    for col in df.columns:
        if col in COLUMN_NAME_MAPPING:
            new_columns[col] = COLUMN_NAME_MAPPING[col]
        else:
            # Fallback normalization
            cleaned = (
                col.strip()
                .lower()
                .replace(" ", "_")
                .replace("(", "_")
                .replace(")", "")
                .replace("/", "_")
                .replace("°", "")
                .replace("", "")
                .replace("²", "2")
            )
            new_columns[col] = cleaned

    df_clean = df.rename(columns=new_columns)
    return df_clean


def inspect_and_handle_outliers(df: pd.DataFrame, numeric_cols: list[str]) -> Dict[str, int]:
    """
    Applies the Interquartile Range (IQR) method to detect extreme sensor spikes.
    Logs findings without blindly deleting physical blast vibration events.
    """
    outlier_counts = {}
    for col in numeric_cols:
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
        if len(outliers) > 0:
            print(f"    - '{col}': {len(outliers)} values outside [{lower_bound:.2f}, {upper_bound:.2f}] (retained as valid physical events)")
    return outlier_counts


def preprocess_dataset(
    raw_file_path: str = "data/raw/ground_vibration_dataset.csv",
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42,
    store_in_sql: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes the full preprocessing pipeline:
    1. Loads raw CSV
    2. Standardizes column names
    3. Handles missing values & duplicates
    4. Maps target labels to standardized risk categories
    5. Inspects outliers
    6. Stores in SQLite database
    7. Performs stratified train/test split
    8. Exports processed CSV files

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (df_all, df_train, df_test)
    """
    print("\n" + "=" * 80)
    print("EXECUTING DATA PREPROCESSING PIPELINE")
    print("=" * 80)

    # 1. Load Raw Data
    raw_path = Path(raw_file_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_path.resolve()}")

    df = pd.read_csv(raw_path)
    print(f"[PREPROCESS] Loaded raw data: {df.shape[0]:,} rows, {df.shape[1]} columns.")

    # 2. Standardize Column Names
    df = clean_column_names(df)
    print(f"[PREPROCESS] Standardized column names: {list(df.columns)}")

    # 3. Handle Duplicates
    initial_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_rows - len(df)
    print(f"[PREPROCESS] Duplicate rows removed: {duplicates_removed}")

    # 4. Handle Missing Values
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Impute numeric features with median
    for num_col in numeric_features:
        if df[num_col].isnull().sum() > 0:
            median_val = df[num_col].median()
            df[num_col] = df[num_col].fillna(median_val)
            print(f"[PREPROCESS] Imputed numeric '{num_col}' with median: {median_val}")

    # Impute categorical features with mode
    for cat_col in categorical_features:
        if df[cat_col].isnull().sum() > 0:
            mode_val = df[cat_col].mode()[0]
            df[cat_col] = df[cat_col].fillna(mode_val)
            print(f"[PREPROCESS] Imputed categorical '{cat_col}' with mode: {mode_val}")

    # 5. Target Standardization & Mapping
    if "vibration_level" in df.columns:
        df["risk_label"] = df["vibration_level"].map(RISK_LEVEL_MAPPING)
        # Handle unmapped labels if any
        df["risk_label"] = df["risk_label"].fillna("WARNING")
        print("\n[PREPROCESS] Standardized Risk Level Distribution:")
        print(df["risk_label"].value_counts())
    else:
        df["risk_label"] = "NORMAL"

    # 6. Outlier Inspection
    print("\n[PREPROCESS] Outlier Detection Summary (IQR Method):")
    inspect_and_handle_outliers(df, numeric_features)

    # 7. Store in SQLite Database
    if store_in_sql:
        print("\n[PREPROCESS] Persisting processed dataset into SQLite database...")
        insert_sensor_data(df)

    # 8. Train/Test Split (Stratified)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["risk_label"]
    )
    print(f"\n[PREPROCESS] Train/Test Split (Stratified 80/20):")
    print(f"    Training samples: {len(df_train):,} ({len(df_train)/len(df)*100:.1f}%)")
    print(f"    Testing samples:  {len(df_test):,} ({len(df_test)/len(df)*100:.1f}%)")

    # 9. Save Processed Datasets
    all_path = out_dir / "processed_all.csv"
    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"

    df.to_csv(all_path, index=False)
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    # 10. Save Dedicated Hardware-Aligned Datasets (excluding blast parameters)
    from src.features.vibration_features import HARDWARE_RAW_SENSOR_COLUMNS
    hw_cols = [c for c in HARDWARE_RAW_SENSOR_COLUMNS if c in df.columns]
    if "risk_label" in df.columns:
        hw_cols_with_target = hw_cols + ["risk_label"]
    else:
        hw_cols_with_target = hw_cols

    hw_all_path = out_dir / "hardware_sensor_dataset.csv"
    hw_train_path = out_dir / "hardware_train.csv"
    hw_test_path = out_dir / "hardware_test.csv"

    df[hw_cols_with_target].to_csv(hw_all_path, index=False)
    df_train[hw_cols_with_target].to_csv(hw_train_path, index=False)
    df_test[hw_cols_with_target].to_csv(hw_test_path, index=False)

    print(f"[PREPROCESS] Saved hardware-aligned dataset ({len(hw_cols)} hardware features + target) to:")
    print(f"    - {hw_all_path.resolve()}")
    print(f"    - {hw_train_path.resolve()}")
    print(f"    - {hw_test_path.resolve()}")

    print(f"[SUCCESS] Saved processed datasets to: {out_dir.resolve()}")
    print("=" * 80 + "\n")

    return df, df_train, df_test


if __name__ == "__main__":
    preprocess_dataset()
