"""
===============================================================================
Master Pipeline: run_pipeline.py
Project: Low-Cost Real-Time Mine Subsidence Monitoring & Early Warning System
===============================================================================

EDUCATIONAL OVERVIEW:
--------------------
This is the master orchestrator script for the entire data, machine learning,
and database pipeline. It ties together every modular component in sequence:

Workflow Sequence:
1. STEP 1: Download Kaggle dataset dynamically using `kagglehub`.
2. STEP 2: Automatically inspect dataset files, columns, types, and domain relevance.
3. STEP 3: Preprocess dataset (cleaning, imputation, outlier check, train/test split)
           and store records into SQLite database (`mine_monitoring.db`).
4. STEP 4: Feature engineering (3D resultant acceleration, scaled distance, kinetic energy).
5. STEP 5: Train Random Forest baseline and XGBoost comparison models.
6. STEP 6: Multi-metric safety evaluation (prioritizing Recall on CRITICAL class)
           and serialization of `models/risk_classifier.joblib`.
7. STEP 7: Run a validation inference check on a sample sensor telemetry payload.
8. STEP 8: Display final instructions for launching the FastAPI service.

How to Run:
    python run_pipeline.py
===============================================================================
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_dataset import download_dataset
from src.data.inspect_dataset import find_data_files, inspect_dataset
from src.data.preprocess import preprocess_dataset
from src.features.vibration_features import prepare_ml_features
from src.ml.evaluate import compare_and_select_best_model
from src.ml.predict import predict_risk


def print_banner(title: str):
    """Prints a styled visual step header."""
    print("\n" + "=" * 80)
    print(f" >>> {title.upper()} <<< ")
    print("=" * 80)


def step_1_download() -> list:
    print_banner("Step 1: Downloading Kaggle Dataset")
    files = download_dataset()
    print(f"[OK] Downloaded {len(files)} raw data files.")
    return files


def step_2_inspect():
    print_banner("Step 2: Automated Exploratory Data Inspection")
    files = find_data_files("data/raw")
    if not files:
        raise FileNotFoundError("No raw data files found to inspect.")
    for f in files:
        inspect_dataset(f)
    print("[OK] Dataset inspection complete.")


def step_3_preprocess_and_store():
    print_banner("Step 3: Data Preprocessing & SQL Database Persistence")
    df_all, df_train, df_test = preprocess_dataset()
    print(f"[OK] Processed {len(df_all)} total rows. Persisted in SQLite `mine_monitoring.db`.")
    return df_all, df_train, df_test


def step_4_feature_engineering(df_all):
    print_banner("Step 4: Geotechnical & Vibration Feature Engineering")
    X, y, feature_names = prepare_ml_features(df_all)
    print(f"[OK] Generated {len(feature_names)} features across {len(X)} records.")
    print(f"[INFO] Features: {feature_names}")


def step_5_train_and_evaluate():
    print_banner("Step 5 & 6: Model Training, Dual Evaluation & Selection")
    artifact_bundle, best_model_name, summary = compare_and_select_best_model()
    print(f"[OK] Best Model Selected: '{best_model_name}' saved to models/risk_classifier.joblib.")
    return artifact_bundle, best_model_name, summary


def step_7_test_inference():
    print_banner("Step 7: Validating Real-Time Inference Interface")
    sample_packet = {
        "node_id": "ESP32_NODE_01",
        "acc_x_ms2": 0.08,
        "acc_y_ms2": 0.09,
        "acc_z_ms2": 0.09,
        "temperature_c": 31.5,
        "ppv_mms": 4.35,
        "frequency_hz": 42.0,
        "psd_value": 0.42,
        "geophone_mms": 2.65,
        "seismometer_ms2": 2.10
    }
    result = predict_risk(sample_packet)
    print("\n[INFERENCE VERIFICATION]:")
    print(f"  Predicted Risk: {result['risk_level']}")
    print(f"  Confidence:     {result['confidence'] * 100:.2f}%")
    print(f"  Probabilities:  {result['probabilities']}")
    print(f"  Model Used:     {result['model_used']}")
    if "node_id" in result:
        print(f"  Node ID:        {result['node_id']}")
    print("[OK] Real-time hardware prediction verified successfully.")


def step_8_run_batch_csv_test():
    print_banner("Step 8: Running Batch CSV Test (Realistic Hardware Scenarios)")
    from scripts.test_sensor_csv import run_batch_sensor_csv_test
    run_batch_sensor_csv_test()
    print("[OK] Batch CSV test completed. Predictions exported to data/test/predictions.csv.")


def run_full_pipeline():
    start_time = time.time()
    print("\n" + "#" * 80)
    print("STARTING HARDWARE-ALIGNED MINE SUBSIDENCE MONITORING ML PIPELINE")
    print("#" * 80)

    # Execute all modular steps
    step_1_download()
    step_2_inspect()
    df_all, _, _ = step_3_preprocess_and_store()
    step_4_feature_engineering(df_all)
    step_5_train_and_evaluate()
    step_7_test_inference()
    step_8_run_batch_csv_test()

    elapsed = time.time() - start_time
    print("\n" + "#" * 80)
    print(f"PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("#" * 80)
    print("\nNEXT STEPS:")
    print("1. Start the FastAPI backend server:")
    print("   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
    print("\n2. Access interactive Swagger API documentation:")
    print("   http://localhost:8000/docs")
    print("\n3. Run batch CSV sensor test:")
    print("   python scripts/test_sensor_csv.py")
    print("\n4. Run the automated test suite:")
    print("   pytest tests/test_pipeline.py -v\n")


if __name__ == "__main__":
    run_full_pipeline()
