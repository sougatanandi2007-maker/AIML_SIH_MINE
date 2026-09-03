"""
===============================================================================
Script: scripts/data_quality_report.py
Project: Low-Cost Real-Time Mine Subsidence Monitoring & Early Warning System
===============================================================================

Data quality analysis script that performs:
1. Missing-value analysis
2. Duplicate detection
3. Outlier analysis (IQR + z-score)
4. Unit consistency checks
5. Label consistency checks
6. Feature distribution analysis
7. Correlation analysis
8. Data leakage checks (PPV → target relationship)

Run:
    python scripts/data_quality_report.py
===============================================================================
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from collections import Counter


def print_section(title: str):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def run_data_quality_report(csv_path: str = None) -> dict:
    """
    Runs comprehensive data quality analysis on the raw or processed dataset.
    
    Returns a summary dictionary with findings.
    """
    # Try to find the dataset
    if csv_path is None:
        candidates = [
            PROJECT_ROOT / "data" / "processed" / "processed_all.csv",
            PROJECT_ROOT / "data" / "raw" / "ground_vibration_dataset.csv",
        ]
        for c in candidates:
            if c.exists():
                csv_path = str(c)
                break
        if csv_path is None:
            print("[ERROR] No dataset found. Run run_pipeline.py first to download data.")
            return {}

    df = pd.read_csv(csv_path)
    findings = {}

    print_section("1. DATASET OVERVIEW")
    print(f"  File:    {csv_path}")
    print(f"  Shape:   {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    findings["shape"] = df.shape

    # -------------------------------------------------------------------------
    # 1. MISSING VALUE ANALYSIS
    # -------------------------------------------------------------------------
    print_section("2. MISSING VALUE ANALYSIS")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct.round(2)
    })
    missing_df = missing_df[missing_df["Missing Count"] > 0]
    if len(missing_df) == 0:
        print("  [OK] No missing values detected in any column.")
    else:
        print(missing_df.to_string())
    findings["missing_total"] = int(missing.sum())

    # -------------------------------------------------------------------------
    # 2. DUPLICATE DETECTION
    # -------------------------------------------------------------------------
    print_section("3. DUPLICATE DETECTION")
    n_duplicates = int(df.duplicated().sum())
    print(f"  Exact duplicate rows: {n_duplicates} ({n_duplicates / len(df) * 100:.2f}%)")
    
    # Check near-duplicates on numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        n_near_dup = int(df[numeric_cols].duplicated().sum())
        print(f"  Numeric-only duplicate rows: {n_near_dup}")
    findings["duplicates"] = n_duplicates

    # -------------------------------------------------------------------------
    # 3. OUTLIER ANALYSIS
    # -------------------------------------------------------------------------
    print_section("4. OUTLIER ANALYSIS (IQR Method)")
    sensor_cols = [c for c in numeric_cols if c not in ["id", "created_at"]]
    outlier_summary = {}
    
    for col in sensor_cols:
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        
        # Z-score outliers (|z| > 3)
        z_scores = np.abs((df[col] - df[col].mean()) / (df[col].std() + 1e-10))
        n_zscore = int((z_scores > 3).sum())
        
        outlier_summary[col] = {"iqr_outliers": n_outliers, "zscore_outliers": n_zscore}
        
        if n_outliers > 0 or n_zscore > 0:
            print(f"  {col:<32} IQR outliers: {n_outliers:>4}   |z|>3: {n_zscore:>4}   "
                  f"Range: [{df[col].min():.3f}, {df[col].max():.3f}]")
    
    findings["outliers"] = outlier_summary

    # -------------------------------------------------------------------------
    # 4. UNIT CONSISTENCY CHECKS
    # -------------------------------------------------------------------------
    print_section("5. UNIT CONSISTENCY CHECKS")
    
    # Check expected physical ranges
    range_checks = {
        "acc_x_ms2": (-50, 50, "m/s² — typical accel range for ground vibration"),
        "acc_y_ms2": (-50, 50, "m/s² — typical accel range for ground vibration"),
        "acc_z_ms2": (-50, 50, "m/s² — typical accel range for ground vibration"),
        "ppv_mms": (0, 1000, "mm/s — PPV should be non-negative"),
        "frequency_hz": (0, 500, "Hz — vibration frequency range"),
        "psd_value": (0, 1e6, "energy units — PSD should be non-negative"),
        "geophone_mms": (0, 1000, "mm/s — geophone velocity"),
        "seismometer_ms2": (-50, 50, "m/s² — seismometer acceleration"),
        "temperature_c": (-40, 80, "°C — plausible environmental temperature"),
    }
    
    for col, (lo, hi, desc) in range_checks.items():
        if col in df.columns:
            vals = df[col]
            out_of_range = int(((vals < lo) | (vals > hi)).sum())
            status = "[OK]" if out_of_range == 0 else f"[WARN] {out_of_range} values out of range"
            print(f"  {col:<24} [{lo}, {hi}]  {status}  ({desc})")

    # -------------------------------------------------------------------------
    # 5. LABEL CONSISTENCY CHECKS
    # -------------------------------------------------------------------------
    print_section("6. LABEL CONSISTENCY CHECKS")
    
    target_col = None
    for candidate in ["risk_label", "vibration_level", "Vibration_Level"]:
        if candidate in df.columns:
            target_col = candidate
            break
    
    if target_col:
        label_dist = df[target_col].value_counts()
        print(f"  Target column: '{target_col}'")
        print(f"  Classes: {list(label_dist.index)}")
        print(f"  Distribution:")
        for label, count in label_dist.items():
            print(f"    {label:<12}: {count:>5} ({count / len(df) * 100:.1f}%)")
        
        # Check class balance ratio
        max_class = label_dist.max()
        min_class = label_dist.min()
        imbalance_ratio = max_class / min_class if min_class > 0 else float("inf")
        print(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}")
        findings["label_distribution"] = label_dist.to_dict()
        findings["imbalance_ratio"] = float(imbalance_ratio)
    else:
        print("  [WARN] No target column found.")

    # -------------------------------------------------------------------------
    # 6. FEATURE DISTRIBUTION ANALYSIS
    # -------------------------------------------------------------------------
    print_section("7. FEATURE DISTRIBUTION ANALYSIS")
    print(f"  {'Feature':<28} {'Mean':>10} {'Std':>10} {'Min':>10} {'Median':>10} {'Max':>10} {'Skew':>8}")
    print(f"  {'-'*90}")
    
    for col in sensor_cols:
        vals = df[col]
        print(f"  {col:<28} {vals.mean():>10.3f} {vals.std():>10.3f} "
              f"{vals.min():>10.3f} {vals.median():>10.3f} {vals.max():>10.3f} "
              f"{vals.skew():>8.3f}")

    # -------------------------------------------------------------------------
    # 7. CORRELATION ANALYSIS
    # -------------------------------------------------------------------------
    print_section("8. CORRELATION ANALYSIS")
    
    if len(sensor_cols) >= 2:
        corr_matrix = df[sensor_cols].corr()
        
        # Find highly correlated pairs (|r| > 0.9)
        print("\n  Highly correlated feature pairs (|r| > 0.90):")
        high_corr_pairs = []
        for i in range(len(sensor_cols)):
            for j in range(i + 1, len(sensor_cols)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.90:
                    high_corr_pairs.append((sensor_cols[i], sensor_cols[j], r))
                    print(f"    {sensor_cols[i]:<28} ↔ {sensor_cols[j]:<28} r = {r:+.4f}")
        
        if not high_corr_pairs:
            print("    None found (all |r| < 0.90)")
        findings["high_correlations"] = high_corr_pairs

    # -------------------------------------------------------------------------
    # 8. DATA LEAKAGE CHECKS
    # -------------------------------------------------------------------------
    print_section("9. DATA LEAKAGE ANALYSIS")
    
    # Check PPV → target relationship (the primary leakage concern)
    ppv_col = "ppv_mms" if "ppv_mms" in df.columns else ("PPV(mm/s)" if "PPV(mm/s)" in df.columns else None)
    
    if ppv_col and target_col:
        print(f"\n  Checking if target '{target_col}' is a direct threshold partition of '{ppv_col}':")
        
        # Map labels to a common scheme for analysis
        label_map = {"Low": "NORMAL", "Medium": "WARNING", "High": "CRITICAL",
                     "NORMAL": "NORMAL", "WARNING": "WARNING", "CRITICAL": "CRITICAL"}
        mapped_labels = df[target_col].map(label_map).fillna(df[target_col])
        
        for label in mapped_labels.unique():
            mask = mapped_labels == label
            ppv_vals = df.loc[mask, ppv_col]
            print(f"    {label:<12}: PPV range [{ppv_vals.min():.3f}, {ppv_vals.max():.3f}], "
                  f"mean={ppv_vals.mean():.3f}")
        
        # Check if PPV alone perfectly separates the classes
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score as acc_score
        
        ppv_array = df[ppv_col].values.reshape(-1, 1)
        target_array = mapped_labels.values
        
        dt = DecisionTreeClassifier(max_depth=2, random_state=42)
        dt.fit(ppv_array, target_array)
        ppv_only_acc = acc_score(target_array, dt.predict(ppv_array))
        
        print(f"\n  [WARN] PPV-only Decision Tree (depth=2) Accuracy: {ppv_only_acc * 100:.2f}%")
        
        if ppv_only_acc > 0.98:
            print(f"\n  [ALERT] TARGET LEAKAGE DETECTED:")
            print(f"     The target label '{target_col}' is almost perfectly determined")
            print(f"     by a simple threshold on '{ppv_col}' alone.")
            print(f"     This means features like 'kinetic_energy_proxy' (= 0.5 * PPV²)")
            print(f"     are monotonic transforms of the target-determining variable.")
            print(f"     The model learns: 'if PPV > threshold → CRITICAL'.")
            print(f"     This is useful for threshold alerting but is NOT genuinely")
            print(f"     predictive from independent sensor signals.")
            findings["target_leakage_detected"] = True
            findings["ppv_only_accuracy"] = float(ppv_only_acc)
        else:
            print(f"  PPV alone does not perfectly separate classes.")
            findings["target_leakage_detected"] = False
    
    # Check for kinetic_energy_proxy leakage
    if "kinetic_energy_proxy" in df.columns and ppv_col:
        expected_ke = 0.5 * df[ppv_col] ** 2
        ke_match = np.allclose(df["kinetic_energy_proxy"], expected_ke, rtol=1e-3)
        if ke_match:
            print(f"\n  [WARN] 'kinetic_energy_proxy' is exactly 0.5 * PPV²")
            print(f"     This is a monotonic transform of the target-determining variable.")

    # Check for train/test data leakage (if split files exist)
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    if train_path.exists() and test_path.exists():
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        # Check for overlapping rows
        train_set = set(df_train.apply(lambda r: tuple(r), axis=1))
        test_set = set(df_test.apply(lambda r: tuple(r), axis=1))
        overlap = train_set & test_set
        print(f"\n  Train/Test overlap check: {len(overlap)} shared rows")
        if len(overlap) > 0:
            print(f"  [ALERT] DATA CONTAMINATION: {len(overlap)} rows appear in both train and test!")
            findings["train_test_overlap"] = len(overlap)
        else:
            print(f"  [OK] No train/test overlap detected.")

    print_section("DATA QUALITY REPORT COMPLETE")
    return findings


if __name__ == "__main__":
    findings = run_data_quality_report()
    if findings:
        print("\n[SUMMARY]")
        print(f"  Dataset shape: {findings.get('shape', 'N/A')}")
        print(f"  Missing values: {findings.get('missing_total', 'N/A')}")
        print(f"  Duplicates: {findings.get('duplicates', 'N/A')}")
        print(f"  Target leakage: {'YES' if findings.get('target_leakage_detected') else 'No'}")
        if findings.get("ppv_only_accuracy"):
            print(f"  PPV-only accuracy: {findings['ppv_only_accuracy'] * 100:.1f}%")
