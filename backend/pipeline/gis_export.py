"""GeoJSON export helpers for model prediction outputs."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def predictions_to_geojson(
    predictions: Iterable[Mapping[str, Any]],
    *,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
) -> dict[str, Any]:
    """Convert prediction records with coordinates into a GeoJSON FeatureCollection."""
    features: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        try:
            latitude = float(prediction[latitude_column])
            longitude = float(prediction[longitude_column])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Prediction {index} must include numeric {latitude_column!r} and "
                f"{longitude_column!r} values."
            ) from error

        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError(f"Prediction {index} has invalid coordinates: {latitude}, {longitude}")
        if not (math.isfinite(latitude) and math.isfinite(longitude)):
            raise ValueError(f"Prediction {index} has non-finite coordinates.")

        properties = {
            key: value
            for key, value in prediction.items()
            if key not in {latitude_column, longitude_column}
        }
        features.append(
            {
                "type": "Feature",
                "id": prediction.get("event_id", index),
                "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
                "properties": properties,
            }
        )

    return {"type": "FeatureCollection", "features": features}


def export_predictions_geojson(
    predictions: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
) -> Path:
    """Persist prediction records as UTF-8 GeoJSON and return the output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    geojson = predictions_to_geojson(
        predictions,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
    )
    path.write_text(json.dumps(geojson, indent=2, default=str) + "\n", encoding="utf-8")
    return path
