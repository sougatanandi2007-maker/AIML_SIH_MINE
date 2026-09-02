# Low-Cost Real-Time Mine Subsidence & Ground Vibration Early Warning System
### Hardware-Aligned Backend & Machine Learning Pipeline (SIH Prototype)

---

## 1. Project Overview & Hardware Context

This project delivers a **hardware-aligned, modular machine learning and backend pipeline** for a Smart India Hackathon (SIH) prototype:

**"Low-Cost Real-Time Mine Subsidence Monitoring & Early Warning System"**

### The Core Problem Solved by this Refactor
In early ML prototypes, models often incorporate operational blast parameters (such as `charge_weight_kg`, `burden_m`, `spacing_m`, and `delay_ms`). However, **real physical ESP32 sensor nodes deployed on mine slopes, pillars, or tailings dams will NOT have access to blast schedules, burden distances, or explosive weights**.

To eliminate **train-serving mismatch and data leakage**, this pipeline retrains and evaluates ML models using **exclusively physical sensor measurements** measurable by edge hardware, backed by an ORM database and a production-ready **FastAPI** service.

```
+--------------------------------------------------------------------------------------------------+
|                            SIH PROTOTYPE HARDWARE-ALIGNED ARCHITECTURE                           |
+--------------------------------------------------------------------------------------------------+

   [ Ground Sensor Nodes ]                [ Miner Helmet ]
   - ESP32 Microcontroller               - ESP32 Edge Node
   - MPU6050 / ADXL355 Accel             - Buzzer / Vibration Motor
   - Geophone Velocity Sensor            - Local Threshold Fallback (Phase 13)
   - DS18B20 Temp Sensor
              │                                  ▲
              │ (LoRa 868/915 MHz Packets)       │ (LoRa Alert Broadcast)
              ▼                                  │
      [ LoRa Gateway ] ──────────────────────────┤
      (Receives RF & forwards via Wi-Fi/4G/MQTT) │
              │                                  │
              ▼ (HTTP POST /predict)             │
   +─────────────────────────────────────────────┴─────────────────────────────────────────────────+
   | CENTRAL BACKEND & ML SERVER (FastAPI)                                                         |
   |                                                                                               |
   |  1. Ingestion: POST /predict (Validates physical sensor telemetry; handles optional fields)   |
   |  2. Feature Engineering: Computes 3D Resultant Shock, Shear Force, Kinetic Energy Proxy       |
   |  3. Preprocessing: StandardScaler pipeline fitted exclusively on 14 hardware features         |
   |  4. Inference: Random Forest / XGBoost calibrated class probability distribution              |
   |  5. Decision: NORMAL / WARNING / CRITICAL                                                     |
   |  6. Database: Relational logging in SQLite (mine_monitoring.db) via SQLAlchemy ORM            |
   +───────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 2. Dataset Understanding & Scientific Honesty

### A. What Dataset is Being Used?
We utilize the Kaggle dataset: `ziya07/multimodal-sensor-fusion-dataset` (`ground_vibration_dataset.csv`, 1,000 samples × 18 columns).

### B. What Does the Dataset Actually Represent?
This dataset contains **blast-induced ground motion telemetry** recorded across surface geophones, seismometers, and accelerometers.

### C. Target Label Reality & Empirical Thresholds
Our exploratory data inspection revealed that the dataset's `Vibration_Level` target is directly and mathematically partitioned by **Peak Particle Velocity (PPV)**:
* **`Low` (`NORMAL`)**: $0.50 \le \text{PPV} < 2.00\text{ mm/s}$ (mean: $1.24\text{ mm/s}$) $\rightarrow$ Safe operational ground motion.
* **`Medium` (`WARNING`)**: $2.00 \le \text{PPV} < 4.00\text{ mm/s}$ (mean: $2.98\text{ mm/s}$) $\rightarrow$ Elevated vibration; threshold advisory.
* **`High` (`CRITICAL`)**: $4.00 \le \text{PPV} \le 5.00\text{ mm/s}$ (mean: $4.52\text{ mm/s}$) $\rightarrow$ Severe dynamic shock; structural hazard.

### D. Crucial Scientific Distinction
> [!IMPORTANT]
> **Vibration Severity Trigger vs. Geological Mine Subsidence:**
> * **Current Prototype**: A calibrated **Ground Vibration & Dynamic Shock Classifier** that alerts mining staff when dynamic ground motion exceeds safe geotechnical thresholds.
> * **Actual Subsidence Prediction**: Geological subsidence (the slow, cumulative vertical compaction and sagging of strata into underground mining voids) requires continuous **InSAR satellite radar geodesy, borehole extensometer displacement logs, and long-term geological strata modeling**.
> * **Scientific Stance**: This model must **NOT** be represented as a validated predictor of catastrophic mine collapse without calibration against real borehole geotechnical sensors.

---

## 3. Hardware Feature Matrix

The retrained ML model operates on a **14-dimensional feature vector** composed of 9 physical sensor measurements and 5 derived geomechanical features.

### A. Physical Sensor Channels (9 Direct Features)
| Feature Name | Physical Sensor Component | Units | Geotechnical & Operational Relevance |
| :--- | :--- | :---: | :--- |
| `acc_x_ms2` | MPU-6050 / ADXL355 Accel | $\text{m/s}^2$ | Dynamic vibration acceleration along X axis. |
| `acc_y_ms2` | MPU-6050 / ADXL355 Accel | $\text{m/s}^2$ | Dynamic vibration acceleration along Y axis. |
| `acc_z_ms2` | MPU-6050 / ADXL355 Accel | $\text{m/s}^2$ | Dynamic vibration acceleration along vertical Z axis. |
| `ppv_mms` | Velocity Geophone / Integrated Accel | $\text{mm/s}$ | Peak Particle Velocity (standard USBM metric for rock damage). |
| `frequency_hz` | Onboard FFT / Zero-Crossing | $\text{Hz}$ | Dominant vibration oscillation frequency. |
| `psd_value` | Onboard Edge FFT | $\text{energy}$ | Power Spectral Density peak energy concentration. |
| `geophone_mms` | Low-cost Velocity Geophone | $\text{mm/s}$ | Ground particle velocity. |
| `seismometer_ms2`| Low-frequency Seismometer / IMU | $\text{m/s}^2$ | Deep ground motion wave amplitude. |
| `temperature_c` | DS18B20 / BMP280 Sensor | $^\circ\text{C}$ | Rock mass and ambient temperature. |

---

### B. Derived Physics Features (5 Engineered Features)
All 5 features are derived strictly from the available hardware telemetry without external metadata:

#### 1. 3D Resultant Dynamic Acceleration (`vibration_magnitude_ms2`)
$$a_{\text{resultant}} = \sqrt{a_x^2 + a_y^2 + a_z^2}$$
* **Meaning**: Total 3D dynamic shock vector magnitude, invariant to sensor mounting tilt.

#### 2. Horizontal Dynamic Shear Acceleration (`vibration_horizontal_ms2`)
$$a_{\text{horizontal}} = \sqrt{a_x^2 + a_y^2}$$
* **Meaning**: Lateral dynamic shear force responsible for mine bench slope slippage and pillar shear.

#### 3. Dynamic Ground Kinetic Energy Proxy (`kinetic_energy_proxy`)
$$E_k = \frac{1}{2} \cdot \text{PPV}^2$$
* **Meaning**: Kinetic energy density transmitted into rock strata by shock wavefronts. Accounts for ~48% of model importance.

#### 4. Dynamic Wave Ratio (`accel_to_velocity_ratio`)
$$\text{Ratio} = \frac{|a_{\text{seismo}}|}{|v_{\text{geo}}| + 10^{-4}}$$
* **Meaning**: Ratio of ground acceleration to particle velocity indicating wavefront steepness.

#### 5. Spectral Power Product (`spectral_power_product`)
$$\text{SPP} = \text{PSD} \cdot f_{\text{dominant}}$$
* **Meaning**: Frequency-domain energy concentration at resonant frequencies.

---

### C. Planned Hardware Features (Currently Unavailable in Kaggle Dataset)
The following physical sensors are part of our future SIH hardware architecture, but are **not present in the Kaggle CSV**:
* `tilt_x_deg`, `tilt_y_deg` (Inclinometer pitch/roll)
* `displacement_mm` (Crack/borehole linear displacement)
* `crack_width_mm` (Extensometer crack width)
* `humidity_pct`, `pressure_hpa` (Environmental)

> [!NOTE]
> **Forward-Compatible Architecture:**
> Rather than fabricating fake historical data, our FastAPI `POST /predict` schema accepts these fields as **optional parameters** (`tilt_x_deg: Optional[float] = None`). When the physical hardware is deployed, the server receives and logs them to the database without breaking model inference.

---

## 4. Machine Learning Models & Evaluation

Both models were retrained exclusively on the 14 hardware features using a stratified 80/20 train/test split.

### A. Random Forest (Baseline)
* **Algorithm**: Ensemble of 150 decorrelated decision trees using bootstrap aggregation (Bagging) and balanced class weighting.
* **Strengths**: Robust to collinearity between triaxial accelerometer axes; immune to outliers.

### B. XGBoost (Comparison)
* **Algorithm**: Gradient Boosted Trees sequentially minimizing multiclass log-loss with second-order Taylor expansions and L1/L2 regularization.
* **Objective**: `multi:softprob` outputting calibrated class probabilities.

### C. Safety-Oriented Evaluation Benchmark

$$\text{Recall}_{\text{CRITICAL}} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

| Model | Overall Accuracy | Macro Precision | Macro Recall | Macro F1 | **CRITICAL Class Recall** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **100.00%** |
| **XGBoost** | 98.00% | 0.9815 | 0.9848 | 0.9827 | **100.00%** |

* **Selection**: Random Forest was selected and exported to `models/risk_classifier.joblib` (and `models/best_model.joblib`), achieving **100% sensitivity on critical early warnings**.

---

## 5. Directory Structure

```
mine_subsidence_ml/
├── data/
│   ├── raw/                           # Raw downloaded Kaggle CSV
│   ├── processed/
│   │   ├── hardware_sensor_dataset.csv # 9 raw hardware features + target
│   │   ├── hardware_train.csv         # Training partition
│   │   ├── hardware_test.csv          # Testing partition
│   │   └── processed_all.csv          # Full processed dataset
│   └── test/
│       ├── hardware_test_data.csv     # Simulated ESP32 test scenarios (Stable, Warning, Critical)
│       └── predictions.csv            # Batch inference output with probabilities
├── models/
│   ├── risk_classifier.joblib         # Serialized hardware model bundle (Pipeline + Encoders)
│   ├── best_model.joblib              # Standalone classifier estimator
│   ├── preprocessor.joblib            # Standalone StandardScaler pipeline
│   ├── feature_names.json             # 14 hardware feature names list
│   └── legacy_full_model.joblib       # Preserved legacy 22-feature backup
├── src/
│   ├── data/
│   │   ├── download_dataset.py        # KaggleHub downloader
│   │   ├── inspect_dataset.py         # Automated exploratory data inspector
│   │   └── preprocess.py              # Hardware dataset isolation & SQLite persistence
│   ├── features/
│   │   └── vibration_features.py      # Hardware physics derivations & waveform signal suite
│   ├── ml/
│   │   ├── train_random_forest.py     # Hardware Random Forest trainer
│   │   ├── train_xgboost.py           # Hardware XGBoost trainer
│   │   ├── evaluate.py                # Dual model comparison & joblib export
│   │   └── predict.py                 # Real-time inference engine with calibrated probabilities
│   ├── database/
│   │   ├── models.py                  # SQLAlchemy 2.0 ORM schema
│   │   └── database.py                # SQLite connection pool & query helpers
│   └── api/
│       └── main.py                    # FastAPI REST service (/predict, /health)
├── scripts/
│   └── test_sensor_csv.py             # Batch CSV telemetry inference tester
├── tests/
│   └── test_pipeline.py               # 5-test pytest suite (100% passing)
├── requirements.txt                   # Minimal required dependencies
├── README.md                          # Comprehensive manual & geomechanics guide
└── run_pipeline.py                    # Master end-to-end pipeline runner
```

---

## 6. How to Run and Test

### Step 1: Execute Full Pipeline End-to-End
```bash
python run_pipeline.py
```
*Executes all stages in ~11 seconds: downloads data, isolates hardware features, retrains Random Forest and XGBoost, exports model bundles, and runs batch test verification.*

### Step 2: Run Automated Unit & Integration Tests
```bash
pytest tests/test_pipeline.py -v
```

### Step 3: Run Batch CSV Sensor Ingestion Test
```bash
python scripts/test_sensor_csv.py
```
*Reads `data/test/hardware_test_data.csv`, evaluates Stable, Warning, and Critical scenarios, and writes outputs with confidence scores to `data/test/predictions.csv`.*

### Step 4: Start the FastAPI Backend Service
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 7. Testing Through Swagger & cURL

### A. Interactive Swagger UI
Open your browser to: **[http://localhost:8000/docs](http://localhost:8000/docs)**

### B. Health Readiness Check (`GET /health`)
```bash
curl http://localhost:8000/health
```
**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "Random Forest",
  "features_count": 14,
  "architecture": "Hardware-Aligned (ESP32/LoRa Compatible)"
}
```

### C. Hardware Telemetry Risk Prediction (`POST /predict`)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "node_id": "ESP32_SLOPE_NODE_01",
       "acc_x_ms2": 0.08,
       "acc_y_ms2": 0.09,
       "acc_z_ms2": 0.09,
       "temperature_c": 31.5,
       "ppv_mms": 4.35,
       "frequency_hz": 42.0,
       "psd_value": 0.42,
       "geophone_mms": 2.65,
       "seismometer_ms2": 2.10,
       "tilt_x_deg": 0.95,
       "displacement_mm": 2.8
     }'
```
**Response:**
```json
{
  "risk_level": "CRITICAL",
  "confidence": 0.9607,
  "probabilities": {
    "CRITICAL": 0.9607,
    "NORMAL": 0.0037,
    "WARNING": 0.0356
  },
  "model_used": "Random Forest",
  "node_id": "ESP32_SLOPE_NODE_01",
  "timestamp": "2026-09-02T19:14:26.185736+00:00"
}
```

---

## 8. ESP32 / LoRa Architecture & Helmet Deployment (Phases 12 & 13)

### Telemetry Packet Flow
1. **Sensor Node (ESP32)**: Samples accelerometers and geophones, calculates PPV, dominant frequency, and peak acceleration, and formats a compact LoRa payload.
2. **LoRa Gateway**: Receives 868/915 MHz RF packets, translates them to JSON, and executes an HTTP `POST` to `/predict`.
3. **Central Server**: Evaluates risk via Random Forest. If `risk_level == "CRITICAL"`, sends an immediate siren trigger and LoRa alert broadcast back to miner helmets.
4. **Local Helmet Fallback (Phase 13)**: To safeguard miners in deep drift tunnels where LoRa gateway connectivity might drop, the helmet ESP32 runs a hardcoded emergency threshold trigger ($a_{\text{res}} > 2.0\text{ m/s}^2$ or continuous rapid tilt) to activate the helmet buzzer autonomously without waiting for server response.
