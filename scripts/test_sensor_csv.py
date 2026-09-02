"""
===============================================================================
Script: scripts/test_sensor_csv.py
Project: Low-Cost Real-Time Mine Subsidence Monitoring & Early Warning System
===============================================================================

EDUCATIONAL OVERVIEW:
--------------------
This utility tests the hardware-aligned ML model using batch CSV telemetry files
simulating ESP32 LoRa sensor node transmissions.

Workflow:
1. Loads simulated edge telemetry from `data/test/hardware_test_data.csv`.
2. Passes each sensor packet through the production `predict_risk()` inference engine.
3. Automatically derives physical features on-the-fly (3D resultant acceleration,
   horizontal shear acceleration, dynamic kinetic energy proxy, wave ratio).
4. Records class probabilities (`NORMAL`, `WARNING`, `CRITICAL`) and confidence scores.
5. Saves enriched results to `data/test/predictions.csv`.
6. Prints a clean terminal summary table comparing scenarios and model outputs.

How to Run:
    python scripts/test_sensor_csv.py
===============================================================================
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.ml.predict import get_predictor

INPUT_CSV_PATH = PROJECT_ROOT / "data" / "test" / "hardware_test_data.csv"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "test" / "predictions.csv"


def run_batch_sensor_csv_test(
    input_path: Path = INPUT_CSV_PATH,
    output_path: Path = OUTPUT_CSV_PATH
) -> pd.DataFrame:
    """
    Reads hardware sensor CSV, executes model predictions row-by-row,
    and outputs results to CSV.
    """
    print("=" * 85)
    print("BATCH HARDWARE SENSOR CSV INFERENCE TEST")
    print("=" * 85)

    if not input_path.exists():
        raise FileNotFoundError(f"Test input CSV not found at: {input_path.resolve()}")

    df_raw = pd.read_csv(input_path)
    print(f"[INFO] Loaded {len(df_raw)} sensor telemetry packets from: {input_path.name}\n")

    predictor = get_predictor()

    results = []
    print(f"{'Node ID':<14} {'Scenario Description':<36} {'Predicted Risk':<16} {'Confidence':<12}")
    print("-" * 85)

    for idx, row in df_raw.iterrows():
        row_dict = row.to_dict()
        pred = predictor.predict(row_dict)

        record = dict(row_dict)
        record["predicted_risk"] = pred["risk_level"]
        record["confidence"] = pred["confidence"]
        record["prob_NORMAL"] = pred["probabilities"].get("NORMAL", 0.0)
        record["prob_WARNING"] = pred["probabilities"].get("WARNING", 0.0)
        record["prob_CRITICAL"] = pred["probabilities"].get("CRITICAL", 0.0)
        record["model_used"] = pred["model_used"]

        results.append(record)

        node = str(row.get("node_id", f"NODE_{idx}"))
        scenario = str(row.get("scenario_description", "Telemetry observation"))[:34]
        risk = pred["risk_level"]
        conf = f"{pred['confidence'] * 100:.1f}%"

        print(f"{node:<14} {scenario:<36} {risk:<16} {conf:<12}")

    df_out = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)

    print("-" * 85)
    print(f"[SUCCESS] All predictions stored to: {output_path.resolve()}")
    print("=" * 85 + "\n")
    return df_out


if __name__ == "__main__":
    run_batch_sensor_csv_test()
