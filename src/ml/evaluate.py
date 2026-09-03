"""
===============================================================================
Module: src/ml/evaluate.py
Project: Low-Cost Real-Time Mine Subsidence Monitoring & Early Warning System
===============================================================================

EDUCATIONAL OVERVIEW:
--------------------
Model evaluation in safety-critical systems (like underground mine monitoring and
slope stability) requires a fundamentally different philosophy than standard ML.

Why Accuracy Alone is Dangerous in Mine Safety:
- Accuracy measures total correct predictions across all classes.
- If 90% of days are 'NORMAL' and 10% are 'CRITICAL', a dumb model predicting
  'NORMAL' 100% of the time achieves 90% accuracy, but FAILS to predict any mine collapse.
- False Negatives in the 'CRITICAL' class cost human lives.
- Therefore, our model selection criteria prioritizes:
  1. Primary Metric: Recall on 'CRITICAL' class (Sensitivity / True Positive Rate).
     Recall = TP / (TP + FN). High recall guarantees we never miss a dangerous vibration spike.
  2. Secondary Metric: Macro F1-Score (harmonic mean of precision and recall across all classes).
  3. Tertiary Metric: Overall test accuracy.

Artifact Persistence with Joblib:
- We serialize not just the raw classifier, but the complete end-to-end bundle:
  - Scikit-learn Pipeline (ColumnTransformer + StandardScaler + OneHotEncoder + Estimator)
  - LabelEncoder (mappings between integer indices and 'NORMAL', 'WARNING', 'CRITICAL')
  - Raw and engineered feature schemas
  - Evaluation diagnostics and confusion matrices
- Destination: `models/risk_classifier.joblib`
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
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.ml.train_random_forest import train_random_forest
from src.ml.train_xgboost import train_xgboost


def compare_and_select_best_model(
    model_output_path: str = "models/risk_classifier.joblib"
) -> Tuple[Any, str, Dict[str, Any]]:
    """
    Trains both Random Forest and XGBoost, presents a side-by-side comparative
    evaluation, selects the optimal safety-first model, and exports the bundle.

    Returns:
        Tuple[Any, str, Dict[str, Any]]: (winning_bundle, winning_model_name, comparative_summary)
    """
    print("\n" + "=" * 80)
    print("STARTING DUAL-MODEL COMPARISON & SAFETY-ORIENTED SELECTION")
    print("=" * 80)

    # 1. Train Random Forest Baseline
    rf_pipeline, rf_encoder, rf_metrics = train_random_forest()

    # 2. Train XGBoost Comparison
    xgb_pipeline, xgb_encoder, xgb_metrics = train_xgboost()

    # 3. Print Side-by-Side Comparison Table
    print("\n" + "=" * 80)
    print("SAFETY & EARLY WARNING BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model':<18} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'Macro F1':<12} {'Critical Recall':<16}")
    print("-" * 80)
    
    for m in [rf_metrics, xgb_metrics]:
        print(
            f"{m['model_name']:<18} "
            f"{m['accuracy'] * 100:>8.2f}%   "
            f"{m['precision']:>10.4f}  "
            f"{m['recall']:>10.4f}  "
            f"{m['f1']:>10.4f}  "
            f"{m['critical_recall'] * 100:>13.2f}%"
        )
    print("=" * 80)

    # 3b. Print Cross-Validation Comparison
    print("\n" + "-" * 80)
    print("CROSS-VALIDATION RESULTS (Stratified 5-Fold on Training Data)")
    print("-" * 80)
    print(f"{'Model':<18} {'CV Accuracy':<14} {'CV Macro F1':<14} {'CV CRITICAL Recall':<20}")
    print("-" * 80)
    for m in [rf_metrics, xgb_metrics]:
        cv_acc = m.get("cv_accuracy", 0)
        cv_f1 = m.get("cv_f1", 0)
        cv_cr = m.get("cv_critical_recall", 0)
        print(
            f"{m['model_name']:<18} "
            f"{cv_acc * 100:>10.2f}%   "
            f"{cv_f1:>12.4f}  "
            f"{cv_cr * 100:>16.2f}%"
        )
    print("-" * 80)

    # 3c. Data Leakage Warning
    print("\n[SCIENTIFIC NOTE]")
    print("  [WARNING] The target label 'Vibration_Level' is a direct threshold partition of PPV.")
    print("    'kinetic_energy_proxy' (= 0.5 * PPV^2) is a monotonic transform of the target-")
    print("    determining variable. Very high accuracy is expected but does NOT indicate")
    print("    genuine predictive capability from independent sensor signals.")
    print("    This model functions as a calibrated threshold-based alerting system.")

    # 4. Model Selection Logic (Safety-First)
    # Primary: critical_recall -> Secondary: f1 -> Tertiary: accuracy
    rf_score = (rf_metrics["critical_recall"], rf_metrics["f1"], rf_metrics["accuracy"])
    xgb_score = (xgb_metrics["critical_recall"], xgb_metrics["f1"], xgb_metrics["accuracy"])

    if rf_score >= xgb_score:
        best_name = "Random Forest"
        best_pipeline = rf_pipeline
        best_encoder = rf_encoder
        best_metrics = rf_metrics
        reason = "Random Forest achieved optimal balance of safety-critical recall and macro F1-score."
    else:
        best_name = "XGBoost"
        best_pipeline = xgb_pipeline
        best_encoder = xgb_encoder
        best_metrics = xgb_metrics
        reason = "XGBoost demonstrated superior gradient-boosted generalization and safety recall."

    print(f"\n[SELECTION] Winning Model: >> {best_name} <<")
    print(f"[SELECTION] Decision Rationale: {reason}")

    # 5. Export Complete Pipeline Artifact
    out_file = Path(model_output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    artifact_bundle = {
        "model_pipeline": best_pipeline,
        "label_encoder": best_encoder,
        "model_name": best_name,
        "classes": list(best_encoder.classes_),
        "feature_names": best_metrics["feature_names"],
        "metrics": best_metrics
    }

    # Primary bundle
    joblib.dump(artifact_bundle, out_file)
    print(f"[SUCCESS] Saved model package to: {out_file.resolve()}")

    # Additional standard artifacts
    best_model_file = out_file.parent / "best_model.joblib"
    preprocessor_file = out_file.parent / "preprocessor.joblib"
    feature_names_file = out_file.parent / "feature_names.json"

    joblib.dump(best_pipeline.named_steps["classifier"], best_model_file)
    joblib.dump(best_pipeline.named_steps["preprocessor"], preprocessor_file)

    import json
    with open(feature_names_file, "w") as f:
        json.dump(best_metrics["feature_names"], f, indent=2)

    print(f"[SUCCESS] Exported best_model.joblib, preprocessor.joblib, and feature_names.json to: {out_file.parent.resolve()}")

    summary = {
        "best_model": best_name,
        "rf_metrics": rf_metrics,
        "xgb_metrics": xgb_metrics,
        "saved_path": str(out_file.resolve())
    }
    return artifact_bundle, best_name, summary


if __name__ == "__main__":
    compare_and_select_best_model()
