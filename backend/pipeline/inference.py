"""Trained CatBoost triage inference (classes 1-5) over the documented 12 features.

Replaces ``inference_stub`` as the default classification path. The artifact and its
input contract live in ``backend/models/artifacts/`` — this module is the only place
that builds the feature row, so retraining touches nothing downstream.

Feature order is load-bearing; ``FEATURE_NAMES`` is asserted against the artifact's own
``feature_names_`` when the model loads, so a retrained model with a different schema
fails loudly instead of silently scoring garbage.

``lulc_class`` categories in the artifact are float-formatted strings ("40.0"), see
``model_lulc_token``. After the model runs, ``landcover.apply_landcover_rules`` zeroes
wildfire/mine/stubble on water or snow; no other post-processing is applied.

The anomaly score is **not** produced here — no Isolation Forest artifact exists in the
repo. It continues to come from the replay pack and is versioned separately so the two
are never conflated.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.models import Detection

from .inference_stub import ANOMALY_MODEL_VERSION, CLASS_NAMES, stub_inference
from .landcover import apply_landcover_rules

logger = logging.getLogger(__name__)

ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "models" / "artifacts" / "catboost_model.cbm"
IFOREST_ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "models" / "artifacts" / "isolation_forest.joblib"
MODEL_VERSION = "catboost-multiclass-13f-0.2.0"

#: Exact training order. Index 9 and 10 are the categorical columns.
FEATURE_NAMES: tuple[str, ...] = (
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
    # Distinct earlier days (1-90) with a detection in the same ~1.1 km cell; separates
    # routine site heat (C5) from a normally quiet site suddenly burning (C1). Live values
    # come from backend/pipeline/recurrence.py.
    "recurrence_days_90d",
)
RECURRENCE_WINDOW_DAYS = 90
CATEGORICAL_INDICES: tuple[int, int] = (9, 10)

#: ESA WorldCover codes the artifact learned. The real-data trainer read ``lulc_class`` as a
#: float column and stringified it, so the model's categories are "40.0", not "40" — a
#: bare "40" silently lands in CatBoost's unknown bucket. ``model_lulc_token`` bridges that.
TRAINED_LULC_CODES = frozenset({"10", "20", "30", "40", "50", "60", "70", "80", "90", "95", "100"})
#: Facility categories the real-data labeler emits (``_stage2_polygon_join.py``).
TRAINED_FACILITY_TYPES = frozenset({
    "general_industrial", "mine_quarry", "power_plant", "brickworks", "factory",
    "other_infrastructure",
})
UNKNOWN_FACILITY = "none"

#: Replay-pack / feed land-cover labels -> ESA WorldCover codes.
LULC_LABEL_TO_CODE: dict[str, str] = {
    "tree_cover": "10",
    "forest": "10",
    "deciduous_forest": "10",
    "evergreen_forest": "10",
    "shrubland": "20",
    "grassland": "30",
    "cropland": "40",
    "agriculture": "40",
    "built_up": "50",
    "industrial_developed": "50",
    "urban": "50",
    "bare_sparse_vegetation": "60",
    "bare_ground": "60",
    "water": "80",
}

# VIIRS 375 m nominal pixel footprint, used when a detection omits scan/track.
DEFAULT_SCAN_KM = 0.375
DEFAULT_TRACK_KM = 0.375


def normalize_lulc(raw: Any) -> str:
    """Map a land-cover label to its ESA WorldCover code, passing codes through."""
    value = str(raw or "").strip()
    if not value:
        return "0"
    try:
        code = str(int(float(value)))
        if code in ("12", "14"):
            return "40"
        return code
    except ValueError:
        return LULC_LABEL_TO_CODE.get(value.lower().replace(" ", "_"), "0")


def model_lulc_token(code: str) -> str:
    """Categorical token the artifact was trained on for a WorldCover code ("40" -> "40.0")."""
    return f"{code}.0" if code in TRAINED_LULC_CODES else code


def normalize_facility_type(raw: Any, *, in_industrial_polygon: bool) -> str:
    """Collapse a facility label onto the artifact's single trained category.

    Descriptive labels ("Petrochemical Refining", "Deciduous Forest Canopy") are unknown
    to the model, so membership of a mapped industrial polygon — not the free-text label
    — decides between ``general_industrial`` and ``none``. A scene outside any industrial
    polygon is never reported as a facility.
    """
    value = str(raw or "").strip().lower().replace(" ", "_")
    if value in TRAINED_FACILITY_TYPES:
        return value
    return "general_industrial" if in_industrial_polygon else UNKNOWN_FACILITY


def _daynight(detection: Detection) -> int:
    """1 = day, 0 = night, from local solar time when the feed omits the flag."""
    if detection.daynight is not None:
        return int(detection.daynight)
    stamp = detection.timestamp
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    utc = stamp.astimezone(timezone.utc)
    solar_hour = (utc.hour + utc.minute / 60.0 + detection.longitude / 15.0) % 24
    return 1 if 6.0 <= solar_hour < 18.0 else 0


def build_feature_row(
    detections: list[Detection], context: dict[str, Any]
) -> tuple[list[Any], bool]:
    """Build one model row for a fused event, in ``FEATURE_NAMES`` order.

    The artifact scores FIRMS-style pixels, while the pipeline reasons about fused
    events, so radiometrics are FRP-weighted means across the event's detections and
    ``frp`` is the event total — the same quantity the rest of the pipeline and the
    counterfactual slider operate on.

    Returns the row and whether its land-cover code is in the trained vocabulary.
    """
    lead = detections[0]
    total_frp = sum(d.frp_mw or 0.0 for d in detections)

    weights = [(d.frp_mw or 0.0) or 1.0 for d in detections]
    weight_sum = sum(weights)

    def weighted(attr: str, fallback: float) -> float:
        values = [(getattr(d, attr) or 0.0, w) for d, w in zip(detections, weights)]
        usable = [(v, w) for v, w in values if v > 0]
        if not usable:
            return fallback
        return sum(v * w for v, w in usable) / sum(w for _, w in usable) if weight_sum else fallback

    ti4 = weighted("bright_ti4_k", 0.0)
    ti5 = weighted("bright_ti5_k", 0.0)

    raw_poly = context.get("is_in_industrial_polygon", False)
    if isinstance(raw_poly, str):
        in_polygon = raw_poly.strip().lower() == "true"
    else:
        in_polygon = bool(raw_poly)

    dist = context.get("distance_to_industrial_m")
    try:
        dist_m = float(dist) if dist is not None and dist != "" else 0.0
    except ValueError:
        dist_m = 0.0
    if dist_m == 0.0 and not in_polygon:
        dist_m = 10000.0

    lulc_code = normalize_lulc(context.get("lulc_class"))
    recurrence = context.get("recurrence_days_90d")
    if recurrence is None and context.get("persistence_score") is not None:
        recurrence = round(float(context["persistence_score"]) * RECURRENCE_WINDOW_DAYS)
    recurrence = max(0, min(RECURRENCE_WINDOW_DAYS, int(recurrence or 0)))

    row: list[Any] = [
        ti4,
        ti5,
        round(ti4 / max(ti5, 1.0), 6),
        round(total_frp, 4),
        lead.scan_km or DEFAULT_SCAN_KM,
        lead.track_km or DEFAULT_TRACK_KM,
        _daynight(lead),
        1 if in_polygon else 0,
        dist_m,
        normalize_facility_type(context.get("facility_type"), in_industrial_polygon=in_polygon),
        model_lulc_token(lulc_code),
        float(context.get("lulc_entropy_500m", 0.0) or 0.0),
        recurrence,
    ]
    return row, lulc_code in TRAINED_LULC_CODES


class IsolationForestInference:
    """Lazy-loaded Isolation Forest for anomaly scoring."""
    def __init__(self, artifact_path: Path = IFOREST_ARTIFACT_PATH) -> None:
        self._artifact_path = artifact_path
        self._model_data = None

    @property
    def available(self) -> bool:
        return self._artifact_path.exists()

    def _load(self) -> Any:
        if self._model_data is not None:
            return self._model_data
        import joblib
        self._model_data = joblib.load(self._artifact_path)
        return self._model_data

    def get_anomaly_score(self, features_dict: dict[str, Any]) -> tuple[float, str]:
        if not self.available:
            return -1.0, ANOMALY_MODEL_VERSION
        try:
            model_data = self._load()
            model = model_data["model"]
            # `feature_names_in_` reflects the column order actually seen by .fit() —
            # trust it over model_data["features"], which is the *intended* order the
            # training script meant to use and can drift out of sync with the real fit
            # order (see backend/models/artifacts/train_isolation_forest.py).
            features = list(getattr(model, "feature_names_in_", model_data["features"]))
            version = model_data["version"]

            import pandas as pd
            row_df = pd.DataFrame([{f: features_dict.get(f, 0.0) for f in features}])
            # Scale decision function to [0, 1] range where higher means more anomalous
            score = float(model.decision_function(row_df)[0])
            anomaly_score = max(0.0, min(1.0, 0.5 - score))
            return anomaly_score, version
        except Exception as e:
            logger.warning(f"Failed to infer anomaly score using Isolation Forest: {e}")
            return -1.0, ANOMALY_MODEL_VERSION

    def get_anomaly_scores(self, features_dicts: list[dict[str, Any]]) -> tuple[list[float], str]:
        """Batch form of ``get_anomaly_score``: one ``decision_function`` call for all rows."""
        if not features_dicts:
            return [], ANOMALY_MODEL_VERSION
        if not self.available:
            return [-1.0] * len(features_dicts), ANOMALY_MODEL_VERSION
        try:
            model_data = self._load()
            model = model_data["model"]
            features = list(getattr(model, "feature_names_in_", model_data["features"]))

            import pandas as pd
            frame = pd.DataFrame([{f: d.get(f, 0.0) for f in features} for d in features_dicts])
            scores = 0.5 - model.decision_function(frame)
            return [max(0.0, min(1.0, float(s))) for s in scores], model_data["version"]
        except Exception as e:
            logger.warning(f"Failed to infer anomaly scores using Isolation Forest: {e}")
            return [-1.0] * len(features_dicts), ANOMALY_MODEL_VERSION


iforest_inference = IsolationForestInference()


class CatBoostInference:
    """Lazy-loaded CatBoost classifier over classes 1-5."""

    def __init__(self, artifact_path: Path = ARTIFACT_PATH) -> None:
        self._artifact_path = artifact_path
        self._model: Any = None
        self._pool: Any = None

    @property
    def available(self) -> bool:
        try:
            self._load()
        except Exception:  # noqa: BLE001 - availability probe must never raise
            return False
        return True

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        from catboost import CatBoostClassifier, Pool  # imported lazily: heavy dependency

        if not self._artifact_path.exists():
            raise FileNotFoundError(f"CatBoost artifact not found: {self._artifact_path}")

        model = CatBoostClassifier()
        model.load_model(str(self._artifact_path))

        names = tuple(model.feature_names_)
        if names != FEATURE_NAMES:
            raise ValueError(
                "CatBoost artifact feature schema does not match the documented contract: "
                f"{names} != {FEATURE_NAMES}"
            )
        classes = [int(c) for c in model.classes_]
        if classes != [1, 2, 3, 4, 5]:
            raise ValueError(f"CatBoost artifact must cover classes 1-5, got {classes}")

        self._pool = Pool
        self._model = model
        return model

    def infer(
        self, detections: list[Detection], context: dict[str, Any], *, live: bool = False
    ) -> dict[str, Any]:
        """``live=True`` (the live single-pixel endpoint) always uses the real Isolation
        Forest score. Curated replay/demo scenarios (``live=False``, the default) always
        honor an explicit ``stub_anomaly_score`` pin from the scenario pack, for
        reproducible demos, falling back to the real model only when nothing is pinned -
        never the other way around, or every curated frame would drift with the model.
        """
        if live:
            # Same code path as the batch live endpoint, so a clicked fire can never get a
            # different answer than the map's bulk classification gave it.
            return self.infer_batch([(detections, context)])[0]
        model = self._load()
        row, lulc_known = build_feature_row(detections, context)
        pool = self._pool([row], cat_features=list(CATEGORICAL_INDICES))
        raw = model.predict_proba(pool)[0]

        model_probabilities = {int(model.classes_[i]): float(p) for i, p in enumerate(raw)}
        features_dict = dict(zip(FEATURE_NAMES, row))

        # One hard physical constraint: no wildfire/mine/stubble on water or snow.
        probabilities, landcover_rule = apply_landcover_rules(
            model_probabilities, normalize_lulc(context.get("lulc_class"))
        )
        top = max(probabilities, key=probabilities.get)

        pinned = context.get("stub_anomaly_score")
        if live or pinned is None:
            anomaly_score, anomaly_version = iforest_inference.get_anomaly_score(features_dict)
            if anomaly_score < 0:
                anomaly_score = max(0.0, min(1.0, float(pinned if pinned is not None else 0.1)))
                anomaly_version = ANOMALY_MODEL_VERSION
        else:
            anomaly_score = max(0.0, min(1.0, float(pinned)))
            anomaly_version = ANOMALY_MODEL_VERSION

        return {
            "class_probabilities": probabilities,
            "class_id": top,
            "class_name": CLASS_NAMES[top],
            "anomaly_score": anomaly_score,
            "model_version": MODEL_VERSION,
            "anomaly_model_version": anomaly_version,
            "features": features_dict,
            "lulc_in_vocabulary": lulc_known,
            "model_class_probabilities": model_probabilities,
            "landcover_rule": landcover_rule,
        }


    def infer_batch(
        self, items: list[tuple[list[Detection], dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Live inference for many events at once: one ``predict_proba`` and one Isolation
        Forest call for the whole batch. Always uses the real anomaly model (``live=True``
        semantics), falling back to a pinned/default score only if that model fails."""
        if not items:
            return []
        model = self._load()
        built = [build_feature_row(detections, context) for detections, context in items]
        rows = [row for row, _ in built]
        raw = model.predict_proba(self._pool(rows, cat_features=list(CATEGORICAL_INDICES)))
        features_dicts = [dict(zip(FEATURE_NAMES, row)) for row in rows]
        anomaly_scores, iforest_version = iforest_inference.get_anomaly_scores(features_dicts)
        classes = [int(c) for c in model.classes_]

        results: list[dict[str, Any]] = []
        for i, (_, context) in enumerate(items):
            model_probabilities = {classes[j]: float(p) for j, p in enumerate(raw[i])}
            probabilities, landcover_rule = apply_landcover_rules(
                model_probabilities, normalize_lulc(context.get("lulc_class"))
            )
            top = max(probabilities, key=probabilities.get)
            anomaly_score, anomaly_version = anomaly_scores[i], iforest_version
            if anomaly_score < 0:
                pinned = context.get("stub_anomaly_score")
                anomaly_score = max(0.0, min(1.0, float(pinned if pinned is not None else 0.1)))
                anomaly_version = ANOMALY_MODEL_VERSION
            results.append({
                "class_probabilities": probabilities,
                "class_id": top,
                "class_name": CLASS_NAMES[top],
                "anomaly_score": anomaly_score,
                "model_version": MODEL_VERSION,
                "anomaly_model_version": anomaly_version,
                "features": features_dicts[i],
                "lulc_in_vocabulary": built[i][1],
                "model_class_probabilities": model_probabilities,
                "landcover_rule": landcover_rule,
            })
        return results


catboost_inference = CatBoostInference()


def get_inference_engine() -> Any:
    """Resolve the active classifier.

    Defaults to the trained CatBoost model, falling back to the hand-authored stub when
    the library or artifact is missing (or when ``JVALYX_INFERENCE=stub`` pins it, which
    is how the golden-scenario demo path is kept deterministic).
    """
    choice = os.getenv("JVALYX_INFERENCE", "catboost").strip().lower()
    if choice == "stub":
        return stub_inference
    if catboost_inference.available:
        return catboost_inference
    logger.warning(
        "CatBoost inference unavailable (install `catboost` and ship %s); "
        "falling back to stub-derived probabilities.",
        ARTIFACT_PATH.name,
    )
    return stub_inference


def active_model_version() -> str:
    """Version string of whichever classifier ``get_inference_engine`` would return."""
    engine = get_inference_engine()
    return getattr(engine, "MODEL_VERSION", MODEL_VERSION if engine is catboost_inference else "stub-0.1.0")
