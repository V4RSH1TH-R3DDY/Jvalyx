import pandas as pd
import numpy as np
import json
from scipy.spatial import cKDTree
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def main():
    print("Loading data...")
    csv_path = "/home/varshith/Downloads/jvalyx_labeled_firms_india_v3.csv"
    geojson_path = "/home/varshith/Downloads/INDIA_ENERGY_PLANTS.geojson"

    df = pd.read_csv(csv_path)

    # Load geojson points
    with open(geojson_path) as f:
        geo_data = json.load(f)

    plants_coords = []
    for feat in geo_data["features"]:
        if feat["geometry"]["type"] == "Point":
            lon, lat = feat["geometry"]["coordinates"]
            plants_coords.append([lat, lon])

    # Build KDTree for fast distance queries
    plants_coords = np.radians(np.array(plants_coords))
    tree = cKDTree(plants_coords)

    print("Calculating distances to nearest energy plants...")
    fires_coords = np.radians(df[["latitude", "longitude"]].values)
    distances, indices = tree.query(fires_coords, k=1)
    
    # Convert radians to meters (Earth radius ~ 6371000m)
    df["distance_to_industrial_m"] = distances * 6371000.0
    
    # Define polygon inclusion based on a 1000m radius of the plant coordinates
    df["is_in_industrial_polygon"] = (df["distance_to_industrial_m"] <= 1000).astype(int)

    # Required features for CatBoost inference
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
        "recurrence_days_90d",
    ]
    CATEGORICAL_FEATURES = ["facility_type", "lulc_class"]

    # Clean data
    df = df.dropna(subset=["class_id"] + FEATURE_NAMES)
    df["class_id"] = df["class_id"].astype(int)
    df = df[df["class_id"].isin([1, 2, 3, 4, 5])]

    df["facility_type"] = df["facility_type"].astype(str)
    df["lulc_class"] = df["lulc_class"].astype(str)

    # RESTRICT CLASSES 1, 3, 5 TO INDUSTRIAL POLYGONS
    mask = ~df["class_id"].isin([1, 3, 5]) | (df["is_in_industrial_polygon"] == 1)
    df = df[mask]

    print("Balancing classes...")
    # Balance classes down to the minority size (with max majority ratio)
    max_majority_ratio = 20
    counts = df["class_id"].value_counts()
    print("Class counts after filtering:")
    print(counts)
    
    minority_count = int(counts.min())
    cap = minority_count * max_majority_ratio

    parts = []
    for cls, group in df.groupby("class_id"):
        parts.append(group.sample(n=cap, random_state=42) if len(group) > cap else group)

    balanced = pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)

    X = balanced[FEATURE_NAMES]
    y = balanced["class_id"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training model on {len(X_train)} rows...")
    model = CatBoostClassifier(
        iterations=450,
        learning_rate=0.07,
        depth=5,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        cat_features=CATEGORICAL_FEATURES,
        class_weights={1: 1.5, 2: 1.0, 3: 4.4, 4: 1.0, 5: 1.2},
        random_seed=42,
        verbose=100,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    preds = model.predict(X_val).flatten()
    print("\nValidation Report:")
    print(classification_report(y_val, preds, digits=3))

    out_path = "/home/varshith/SIH_2026_Jvalyx/backend/models/artifacts/catboost_model.cbm"
    model.save_model(out_path)
    print(f"Saved custom model to {out_path}")

if __name__ == "__main__":
    main()
