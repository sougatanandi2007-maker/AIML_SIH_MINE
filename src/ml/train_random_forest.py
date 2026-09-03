"""
===============================================================================
Module: src/ml/train_random_forest.py
Project: Low-Cost Real-Time Mine Subsidence Monitoring & Early Warning System
===============================================================================

EDUCATIONAL OVERVIEW:
--------------------
Random Forest is our baseline sensor-fusion classifier.

Why Random Forest for Mine Sensor Fusion?
1. Ensemble of Decision Trees: Combines multiple de-correlated decision trees trained
   on bootstrap subsets of data (Bagging) with random feature subspace sampling.
2. Non-Linear Sensor Interactions: Effectively captures complex physical relationships
   (e.g., high PPV + Soft Soil -> higher risk) without assuming linear boundaries.
3. Class Imbalance Handling: Utilizes `class_weight='balanced'` to prevent bias toward
   the majority class and ensure high sensitivity (recall) on critical safety warnings.
4. Robustness: Outliers in blast spikes do not destabilize the tree splitting criteria
   (unlike distance-based models like KNN or SVM).
5. Interpretability: Provides Gini-importance scores to identify which sensor parameters
   (e.g., PPV, resultant acceleration, charge weight) contribute most to risk predictions.

Preprocessing in the Scikit-learn Pipeline:
- Numerical features: Scaled via `StandardScaler` (z = (x - mu) / sigma)
- Categorical features ('soil_type'): Transformed via `OneHotEncoder(handle_unknown='ignore')`
- Target labels ('NORMAL', 'WARNING', 'CRITICAL'): Encoded via `LabelEncoder`

Inputs:
- Processed train & test CSVs from `data/processed/`

Outputs:
- Trained Scikit-Learn Pipeline
- Performance metrics on the unseen test set
===============================================================================
"""

import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.features.vibration_features import prepare_ml_features


def build_preprocessing_pipeline(X: pd.DataFrame) -> Tuple[ColumnTransformer, list[str], list[str]]:
    """
    Constructs a ColumnTransformer that scales numerical features and
    one-hot encodes categorical features if present.
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    transformers = [("num", StandardScaler(), numeric_cols)]
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, numeric_cols, categorical_cols


def train_random_forest(
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv"
) -> Tuple[Pipeline, LabelEncoder, Dict[str, Any]]:
    """
    Trains and validates the baseline Random Forest sensor-fusion model.

    Returns:
        Tuple[Pipeline, LabelEncoder, Dict[str, Any]]: (fitted_pipeline, label_encoder, metrics_dict)
    """
    print("\n" + "=" * 80)
    print("TRAINING BASELINE MODEL: RANDOM FOREST CLASSIFIER")
    print("=" * 80)

    # 1. Load Data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 2. Extract Features and Labels
    X_train, y_train_raw, feature_names = prepare_ml_features(df_train)
    X_test, y_test_raw, _ = prepare_ml_features(df_test)

    # 3. Encode Target Labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    class_names = list(label_encoder.classes_)
    critical_idx = list(label_encoder.classes_).index("CRITICAL") if "CRITICAL" in label_encoder.classes_ else -1
    print(f"[ML] Target Classes Encoded: {dict(zip(range(len(class_names)), class_names))}")

    # 4. Build Pipeline
    preprocessor, num_cols, cat_cols = build_preprocessing_pipeline(X_train)
    
    rf_classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf_classifier)
    ])

    # 5. Train Model
    print(f"[ML] Fitting Random Forest on {len(X_train)} training samples...")
    model_pipeline.fit(X_train, y_train)

    # 5b. Stratified 5-Fold Cross-Validation on Training Data
    print("[ML] Running stratified 5-fold cross-validation on training data...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_predictions = cross_val_predict(model_pipeline, X_train, y_train, cv=cv)
    cv_acc = accuracy_score(y_train, cv_predictions)
    cv_f1 = f1_score(y_train, cv_predictions, average="macro", zero_division=0)
    cv_critical_recall = 0.0
    if critical_idx >= 0:
        cv_critical_recall = recall_score(
            y_train == critical_idx, cv_predictions == critical_idx, zero_division=0
        )
    print(f"    CV Accuracy:          {cv_acc * 100:.2f}%")
    print(f"    CV Macro F1:          {cv_f1:.4f}")
    print(f"    CV CRITICAL Recall:   {cv_critical_recall * 100:.2f}%")

    # 6. Evaluate on Test Set
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    # Critical class recall (high safety priority)
    critical_recall = 0.0
    if critical_idx >= 0:
        critical_recall = recall_score(
            y_test == critical_idx, y_pred == critical_idx, zero_division=0
        )

    cm = confusion_matrix(y_test, y_pred)

    print("\n[ML] --- RANDOM FOREST EVALUATION RESULTS ---")
    print(f"    Overall Accuracy:        {acc * 100:.2f}%")
    print(f"    Macro Precision:         {prec_macro:.4f}")
    print(f"    Macro Recall:            {rec_macro:.4f}")
    print(f"    Macro F1-Score:          {f1_macro:.4f}")
    print(f"    CRITICAL Class Recall:   {critical_recall * 100:.2f}% (Safety Early Warning Sensitivity)")
    print("\n[ML] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    print("[ML] Confusion Matrix:")
    print(cm)

    # 7. Extract Feature Importances
    fitted_preprocessor = model_pipeline.named_steps["preprocessor"]
    fitted_rf = model_pipeline.named_steps["classifier"]
    
    # Get encoded feature names
    encoded_cat_names = []
    if "cat" in fitted_preprocessor.named_transformers_ and cat_cols:
        cat_encoder = fitted_preprocessor.named_transformers_["cat"]
        encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
    all_transformed_features = num_cols + encoded_cat_names
    
    importances = fitted_rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    
    print("\n[ML] Top 10 Most Predictive Sensor Features:")
    for rank, idx in enumerate(top_indices, start=1):
        feat = all_transformed_features[idx] if idx < len(all_transformed_features) else f"Feature_{idx}"
        print(f"    {rank:2d}. {feat:<28} Importance: {importances[idx]:.4f}")

    metrics = {
        "model_name": "Random Forest",
        "accuracy": float(acc),
        "precision": float(prec_macro),
        "recall": float(rec_macro),
        "f1": float(f1_macro),
        "critical_recall": float(critical_recall),
        "cv_accuracy": float(cv_acc),
        "cv_f1": float(cv_f1),
        "cv_critical_recall": float(cv_critical_recall),
        "confusion_matrix": cm.tolist(),
        "classes": class_names,
        "feature_names": feature_names
    }

    print("=" * 80 + "\n")
    return model_pipeline, label_encoder, metrics


if __name__ == "__main__":
    train_random_forest()
