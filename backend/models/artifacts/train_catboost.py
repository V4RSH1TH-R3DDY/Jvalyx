"""Training script for Jvalyx CatBoost multi-class thermal incident classifier.

With ``--csv``, trains on real labeled detections (see ``data/training/`` and the
labeling pipeline that produced it) instead of synthetic data. Real class counts are
heavily imbalanced (agricultural/wildfire detections vastly outnumber industrial/mining/
flare ones), so the real-data path downsamples oversized classes and applies CatBoost's
``SqrtBalanced`` class weighting rather than pretending the classes are naturally equal
size, the way the synthetic generator below does.

Without ``--csv``, falls back to the original synthetic generator — useful only for
smoke-testing that the training pipeline runs end to end. It has no relationship to real
fires and must never be treated as a substitute for real data.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ARTIFACT_PATH = Path(__file__).resolve().parent / "catboost_model.cbm"

FEATURE_NAMES = [
    "bright_ti4",
    "bright_ti5",
    "temp_ratio",
    "frp",
    "scan",
    "track",
    "daynight",
    "is_in_industrial_polygon",
    "distance_to_industrial_m",
    "facility_type",
    "lulc_class",
    "lulc_entropy_500m",
]

CATEGORICAL_FEATURES = ["facility_type", "lulc_class"]

#: Real-data-only features not yet part of the deployed contract in
#: ``backend/pipeline/inference.py``. Included in training automatically when present in
#: the CSV, but shipping a model trained with one requires adding it to that module's
#: ``FEATURE_NAMES``/``build_feature_row`` too, so live inference can compute it — a
#: model trained on a feature the live pipeline never fills in will just see it as 0/NaN
#: at inference time.
OPTIONAL_FEATURES = ["recurrence_count_90d"]


def generate_balanced_training_data(n_per_class: int = 1200) -> tuple[pd.DataFrame, np.ndarray]:
    np.random.seed(42)
    rows = []
    labels = []

    # -------------------------------------------------------------
    # Class 1: Accidental Industrial Fire / Explosion
    # Severe thermal escalation in industrial installation (>70 to 600+ MW)
    # High TI4 brightness (>355-450 K), temp_ratio > 1.20, LULC 50
    # -------------------------------------------------------------
    for _ in range(n_per_class):
        frp = np.random.uniform(70.0, 600.0)
        ti4 = np.random.uniform(355.0, 450.0)
        ti5 = np.random.uniform(294.0, 320.0)
        dist = np.random.choice([0.0, 20.0, 100.0, 250.0])
        rows.append({
            "bright_ti4": ti4,
            "bright_ti5": ti5,
            "temp_ratio": ti4 / ti5,
            "frp": frp,
            "scan": np.random.choice([0.375, 0.38, 0.5, 0.75]),
            "track": np.random.choice([0.375, 0.38, 0.5, 0.75]),
            "daynight": np.random.choice([0, 1]),
            "is_in_industrial_polygon": 1,
            "distance_to_industrial_m": dist,
            "facility_type": "general_industrial",
            "lulc_class": "50",
            "lulc_entropy_500m": np.random.uniform(1.2, 2.2),
        })
        labels.append(1)

    # -------------------------------------------------------------
    # Class 2: Wildfire or Forest Fire
    # Vegetation/canopy fire (LULC 10/tree cover, 20/shrub), outside industrial
    # Wide FRP range (20 to 450 MW), drift & spread
    # -------------------------------------------------------------
    for _ in range(n_per_class):
        frp = np.random.uniform(20.0, 450.0)
        ti4 = np.random.uniform(335.0, 400.0)
        ti5 = np.random.uniform(290.0, 310.0)
        dist = np.random.uniform(3000.0, 60000.0)
        lulc = np.random.choice(["10", "20"], p=[0.8, 0.2])
        rows.append({
            "bright_ti4": ti4,
            "bright_ti5": ti5,
            "temp_ratio": ti4 / ti5,
            "frp": frp,
            "scan": np.random.choice([0.375, 0.38, 0.5, 0.75]),
            "track": np.random.choice([0.375, 0.38, 0.5, 0.75]),
            "daynight": np.random.choice([0, 1]),
            "is_in_industrial_polygon": 0,
            "distance_to_industrial_m": dist,
            "facility_type": "none",
            "lulc_class": lulc,
            "lulc_entropy_500m": np.random.uniform(0.1, 0.75),
        })
        labels.append(2)

    # -------------------------------------------------------------
    # Class 3: Uncontrolled Mining / Coal-Seam Fire
    # Open-cast mine / coalfield (LULC 60 bare ground or 30 scrub/spoil), in or near mine polygon
    # Moderate sustained FRP (18 to 140 MW)
    # -------------------------------------------------------------
    for _ in range(n_per_class):
        frp = np.random.uniform(18.0, 140.0)
        ti4 = np.random.uniform(325.0, 365.0)
        ti5 = np.random.uniform(288.0, 305.0)
        in_poly = np.random.choice([1, 0], p=[0.75, 0.25])
        dist = 0.0 if in_poly else np.random.uniform(50.0, 2500.0)
        fac = "general_industrial" if in_poly else "none"
        lulc = np.random.choice(["60", "30"], p=[0.85, 0.15])
        rows.append({
            "bright_ti4": ti4,
            "bright_ti5": ti5,
            "temp_ratio": ti4 / ti5,
            "frp": frp,
            "scan": np.random.choice([0.375, 0.38, 0.5, 0.75]),
            "track": np.random.choice([0.375, 0.38, 0.5, 0.75]),
            "daynight": np.random.choice([0, 1]),
            "is_in_industrial_polygon": in_poly,
            "distance_to_industrial_m": dist,
            "facility_type": fac,
            "lulc_class": lulc,
            "lulc_entropy_500m": np.random.uniform(0.6, 1.6),
        })
        labels.append(3)

    # -------------------------------------------------------------
    # Class 4: Agricultural / Stubble Burning
    # Cropland (LULC 40) or grassland (30), outside industrial (>2000m)
    # Low to moderate FRP (5 to 45 MW)
    # -------------------------------------------------------------
    for _ in range(n_per_class):
        frp = np.random.uniform(5.0, 45.0)
        ti4 = np.random.uniform(315.0, 345.0)
        ti5 = np.random.uniform(285.0, 300.0)
        dist = np.random.uniform(2000.0, 50000.0)
        lulc = np.random.choice(["40", "30"], p=[0.8, 0.2])
        rows.append({
            "bright_ti4": ti4,
            "bright_ti5": ti5,
            "temp_ratio": ti4 / ti5,
            "frp": frp,
            "scan": np.random.choice([0.375, 0.38, 0.5, 0.75]),
            "track": np.random.choice([0.375, 0.38, 0.5, 0.75]),
            "daynight": np.random.choice([0, 1]),
            "is_in_industrial_polygon": 0,
            "distance_to_industrial_m": dist,
            "facility_type": "none",
            "lulc_class": lulc,
            "lulc_entropy_500m": np.random.uniform(0.2, 0.9),
        })
        labels.append(4)

    # -------------------------------------------------------------
    # Class 5: Persistent Flare / Routine Heat
    # Inside industrial polygon (1), distance 0, general_industrial, LULC 50
    # Normal baseline FRP (12 to 65 MW), hot flare tip, within statistical baseline
    # -------------------------------------------------------------
    for _ in range(n_per_class):
        frp = np.random.uniform(12.0, 65.0)
        ti4 = np.random.uniform(330.0, 355.0)
        ti5 = np.random.uniform(290.0, 302.0)
        rows.append({
            "bright_ti4": ti4,
            "bright_ti5": ti5,
            "temp_ratio": ti4 / ti5,
            "frp": frp,
            "scan": np.random.choice([0.375, 0.38, 0.5, 0.75]),
            "track": np.random.choice([0.375, 0.38, 0.5, 0.75]),
            "daynight": np.random.choice([0, 1]),
            "is_in_industrial_polygon": 1,
            "distance_to_industrial_m": 0.0,
            "facility_type": "general_industrial",
            "lulc_class": "50",
            "lulc_entropy_500m": np.random.uniform(1.2, 2.0),
        })
        labels.append(5)

    df = pd.DataFrame(rows)[FEATURE_NAMES]
    return df, np.array(labels)


def load_real_training_data(
    csv_path: str, max_majority_ratio: int
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load a labeled real-detection CSV and downsample oversized classes.

    ``max_majority_ratio`` caps every class at that multiple of the smallest class's row
    count, picked by uniform random sampling within the class. This trims redundant
    majority-class rows (e.g. near-identical agricultural burns) without fabricating any
    minority-class rows — synthetic oversampling would reintroduce the exact
    made-up-data problem this real-data path exists to fix.
    """
    df = pd.read_csv(csv_path)
    if "class_id" not in df.columns:
        raise ValueError("Training CSV must have a 'class_id' column with values 1-5.")

    df = df.dropna(subset=["class_id"])
    df["class_id"] = df["class_id"].astype(int)
    df = df[df["class_id"].isin([1, 2, 3, 4, 5])]

    feature_cols = list(FEATURE_NAMES) + [c for c in OPTIONAL_FEATURES if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {missing}")

    df = df.dropna(subset=feature_cols)
    df["facility_type"] = df["facility_type"].astype(str)
    df["lulc_class"] = df["lulc_class"].astype(str)

    counts = df["class_id"].value_counts().sort_index()
    print("Raw class counts before downsampling:")
    print(counts.to_string())

    minority_count = int(counts.min())
    cap = minority_count * max_majority_ratio
    parts = []
    for cls, group in df.groupby("class_id"):
        parts.append(group.sample(n=cap, random_state=42) if len(group) > cap else group)
    balanced = pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"\nAfter capping majority classes at {max_majority_ratio}x the smallest class ({minority_count} rows):")
    print(balanced["class_id"].value_counts().sort_index().to_string())

    return balanced[feature_cols], balanced["class_id"].to_numpy(), feature_cols


def train_and_save_model(csv_path: str | None = None, max_majority_ratio: int = 20) -> None:
    if csv_path:
        print(f"Loading real training data from {csv_path}")
        X, y, feature_cols = load_real_training_data(csv_path, max_majority_ratio)
        cat_features = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
        auto_class_weights = "SqrtBalanced"
    else:
        print(
            "WARNING: no --csv given. Training on synthetic np.random data with no "
            "relationship to real fires. Use only to smoke-test the training pipeline."
        )
        X, y = generate_balanced_training_data(n_per_class=1200)
        feature_cols = FEATURE_NAMES
        cat_features = CATEGORICAL_FEATURES
        auto_class_weights = None  # already artificially balanced 1:1

    # Restrict Industrial Escalations (1) and Mining Fires (3) to industrial polygons
    mask = ~np.isin(y, [1, 3]) | (X["is_in_industrial_polygon"] == 1)
    X = X[mask]
    y = y[mask]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(
        f"\nTraining CatBoost on {len(X_train)} rows / {len(feature_cols)} features, "
        f"validating on {len(X_val)}..."
    )
    model = CatBoostClassifier(
        iterations=450,
        learning_rate=0.07,
        depth=5,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        cat_features=cat_features,
        auto_class_weights=auto_class_weights,
        random_seed=42,
        verbose=100,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    preds = model.predict(X_val).flatten()
    print("\nPer-class validation metrics (accuracy alone is meaningless on imbalanced classes):")
    print(classification_report(y_val, preds, digits=3))
    print("Confusion matrix (rows=true, cols=predicted), classes [1,2,3,4,5]:")
    print(confusion_matrix(y_val, preds, labels=[1, 2, 3, 4, 5]))

    print(f"\nSaving model to {ARTIFACT_PATH}...")
    model.save_model(str(ARTIFACT_PATH))
    print("Model successfully saved!")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to a labeled real-detection CSV (must include a class_id column, 1-5). "
        "Omit to fall back to the synthetic smoke-test generator.",
    )
    parser.add_argument(
        "--max-majority-ratio", type=int, default=20,
        help="Cap each class at this multiple of the smallest class's row count (default 20).",
    )
    args = parser.parse_args()
    train_and_save_model(csv_path=args.csv, max_majority_ratio=args.max_majority_ratio)


if __name__ == "__main__":
    main()
