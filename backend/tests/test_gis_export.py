import json

import pytest

from backend.pipeline import export_predictions_geojson, predictions_to_geojson


def test_predictions_to_geojson_uses_geojson_coordinate_order() -> None:
    result = predictions_to_geojson(
        [
            {
                "event_id": "evt-001",
                "latitude": 19.0760,
                "longitude": 72.8777,
                "class_id": 1,
                "anomaly_score": 0.8,
            }
        ]
    )

    assert result["type"] == "FeatureCollection"
    assert result["features"][0]["geometry"] == {
        "type": "Point",
        "coordinates": [72.8777, 19.076],
    }
    assert result["features"][0]["properties"]["class_id"] == 1


def test_export_predictions_geojson_persists_a_valid_file(tmp_path) -> None:
    output_path = export_predictions_geojson(
        [{"event_id": "evt-001", "latitude": 0, "longitude": 0, "class_id": 2}],
        tmp_path / "predictions.geojson",
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.exists()
    assert saved["features"][0]["id"] == "evt-001"


def test_predictions_to_geojson_rejects_invalid_coordinates() -> None:
    with pytest.raises(ValueError, match="invalid coordinates"):
        predictions_to_geojson([{"latitude": 91, "longitude": 0}])