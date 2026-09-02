"""
===============================================================================
Module: src/data/inspect_dataset.py
Project: Low-Cost Real-Time Mine Subsidence Monitoring & Early Warning System
===============================================================================

EDUCATIONAL OVERVIEW:
--------------------
Exploratory Data Analysis (EDA) and automated data inspection are critical before
performing any preprocessing or modeling. Never assume dataset column names, types,
or value ranges.

Key Inspection Steps:
1. Shape & Structure: Total rows and columns in the dataset.
2. Data Types: Distinguishing numeric (floats/ints) from categorical/string columns.
3. Data Integrity: Missing (null/NaN) values and duplicate records.
4. Statistical Summaries: Mean, standard deviation, min, max, percentiles.
5. Domain Categorization: Mapping discovered columns to physical mining/sensor concepts:
   - Ground Motion / Vibration: Acceleration, PPV (Peak Particle Velocity), Frequency
   - Environmental Context: Temperature, Humidity, Wind speed, Rainfall
   - Geological / Soil: Soil type, Rock density, Geological layer
   - Operational / Mining: Blast charge, Distance to blast, Hole depth
   - Target / Labels: Vibration/Risk severity levels

Inputs:
- raw_data_dir (str): Path to directory containing raw CSV files (default: 'data/raw')

Outputs:
- Comprehensive diagnostic printout
- Python dictionary containing inspection metadata for downstream pipeline modules.
===============================================================================
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np


def find_data_files(raw_data_dir: str = "data/raw") -> List[Path]:
    """
    Scans the raw data directory for tabular data files (CSV, TSV, Parquet).
    """
    raw_path = Path(raw_data_dir)
    supported_extensions = ["*.csv", "*.tsv", "*.parquet", "*.json"]
    found_files = []
    for ext in supported_extensions:
        found_files.extend(list(raw_path.glob(ext)))
    return found_files


def inspect_dataset(file_path: Path) -> Dict[str, Any]:
    """
    Performs comprehensive automated exploratory inspection of a dataset file.

    Args:
        file_path (Path): Path to the CSV/tabular file to inspect.

    Returns:
        Dict[str, Any]: Structured summary of dataset properties.
    """
    print("=" * 80)
    print(f"AUTOMATED DATASET INSPECTION: {file_path.name}")
    print("=" * 80)

    # 1. Load Data
    df = pd.read_csv(file_path)
    rows, cols = df.shape

    print(f"\n[1] DATASET SHAPE:")
    print(f"    Total Rows (Samples):    {rows:,}")
    print(f"    Total Columns (Features): {cols}")

    # 2. First 5 Rows Preview
    print(f"\n[2] FIRST 5 ROWS PREVIEW:")
    print(df.head())

    # 3. Column Metadata Inspection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print(f"\n[3] COLUMN SPECIFICATION & DATA INTEGRITY:")
    print(f"{'Column Name':<30} {'Type':<12} {'Missing':<10} {'% Missing':<12} {'Uniques':<10}")
    print("-" * 76)

    col_details = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        null_count = int(df[col].isnull().sum())
        null_pct = (null_count / rows) * 100
        unique_count = int(df[col].nunique())

        print(f"{col:<30} {dtype_str:<12} {null_count:<10} {null_pct:<11.2f}% {unique_count:<10}")
        col_details.append({
            "name": col,
            "dtype": dtype_str,
            "missing": null_count,
            "missing_pct": null_pct,
            "uniques": unique_count
        })

    # 4. Duplicate Check
    duplicate_rows = int(df.duplicated().sum())
    print(f"\n[4] DUPLICATE ROWS CHECK:")
    print(f"    Duplicate count: {duplicate_rows} ({(duplicate_rows / rows) * 100:.2f}%)")

    # 5. Numerical Columns Statistics
    print(f"\n[5] NUMERICAL FEATURES SUMMARY:")
    if numeric_cols:
        print(df[numeric_cols].describe().T[["mean", "std", "min", "50%", "max"]])
    else:
        print("    No numeric columns found.")

    # 6. Categorical Columns & Unique Values
    print(f"\n[6] CATEGORICAL FEATURES & VALUE DISTRIBUTIONS:")
    for col in categorical_cols:
        unique_cnt = df[col].nunique()
        print(f"\n    Column '{col}' ({unique_cnt} unique values):")
        val_counts = df[col].value_counts(dropna=False)
        if unique_cnt > 10:
            print(f"      (Showing top 5 of {unique_cnt} unique values):")
            for val, count in val_counts.head(5).items():
                print(f"        - {val}: {count} ({count / rows * 100:.1f}%)")
        else:
            for val, count in val_counts.items():
                print(f"        - {val}: {count} ({count / rows * 100:.1f}%)")

    # 7. Domain Relevance Mapping
    print(f"\n[7] DOMAIN RELEVANCE & SENSOR CATEGORIZATION:")
    domain_mapping = categorize_columns(df.columns.tolist())
    for category, matched_cols in domain_mapping.items():
        print(f"    {category:<32}: {matched_cols if matched_cols else 'None detected'}")

    print("\n" + "=" * 80)

    return {
        "file_name": file_path.name,
        "rows": rows,
        "columns": cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_rows": duplicate_rows,
        "column_details": col_details,
        "domain_mapping": domain_mapping
    }


def categorize_columns(columns: List[str]) -> Dict[str, List[str]]:
    """
    Heuristically categorizes dataset columns based on standard mining and
    vibration monitoring terminology.
    """
    mapping = {
        "Vibration / Acceleration": [],
        "Frequency / Dynamic Shock": [],
        "PPV (Peak Particle Velocity)": [],
        "Temperature / Environmental": [],
        "Soil / Geological Parameters": [],
        "Blast / Mining Operations": [],
        "Target / Risk / Labels": []
    }

    for col in columns:
        col_lower = col.lower()
        if any(k in col_lower for k in ["risk", "level", "label", "category", "target", "class", "damage", "warning", "severity", "status"]):
            mapping["Target / Risk / Labels"].append(col)
        elif any(k in col_lower for k in ["vibrat", "acc", "seismo", "geophone", "displacement", "tilt"]):
            mapping["Vibration / Acceleration"].append(col)
        elif any(k in col_lower for k in ["freq", "hz", "dominant", "psd"]):
            mapping["Frequency / Dynamic Shock"].append(col)
        elif any(k in col_lower for k in ["ppv", "peak_particle", "velocity"]):
            mapping["PPV (Peak Particle Velocity)"].append(col)
        elif any(k in col_lower for k in ["temp", "humidity", "wind", "rain", "weather", "moisture", "pressure"]):
            mapping["Temperature / Environmental"].append(col)
        elif any(k in col_lower for k in ["soil", "rock", "geo", "strata", "density"]):
            mapping["Soil / Geological Parameters"].append(col)
        elif any(k in col_lower for k in ["blast", "charge", "distance", "hole", "depth", "burden", "spacing", "delay"]):
            mapping["Blast / Mining Operations"].append(col)

    return mapping


if __name__ == "__main__":
    files = find_data_files()
    if not files:
        print("[ERROR] No data files found in data/raw/. Please run src/data/download_dataset.py first.")
    else:
        for f in files:
            inspect_dataset(f)
