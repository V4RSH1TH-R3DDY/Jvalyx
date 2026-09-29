# Cross-layer findings

Discrepancies the backend work surfaced that live in **other layers** (frontend / ML /
docs / infra). Logged here for the owners of those layers — the backend team does not fix
these.

_Last updated: 2026-09-11._

---

## Docs

| # | Finding | Suggested fix |
|---|---|---|
| D1 | `docs/Jvalyx_Comprehensive_Build_Plan.md` was reflowed by a markdown auto-formatter (tables re-padded, trailing double-spaces stripped). It joined several bold intro lines onto one line (`...intelligence product.**Recommended MVP:**`). Not a backend change. | `git checkout -- docs/Jvalyx_Comprehensive_Build_Plan.md`, then disable format-on-save for that file or commit a clean reformat deliberately. |

## Frontend ↔ backend contract mismatches

**Resolved.** The dashboard is wired to the backend; `frontend/src/services/adapters.ts`
is the single mapping layer between the two vocabularies (F5's suggested fix). F1/F2 were
additionally closed at the source — the backend now speaks the frontend's
`SensorAgreementState` vocabulary and the frontend accepts `medium`. Kept below as the
record of what the adapter reconciles:

| # | Frontend (`types/index.ts`) | Backend (`models/schemas.py`) |
|---|---|---|
| F1 | `SensorAgreementState` = `full_agreement \| temporally_confirmed_spatially_coarse \| single_sensor_high_res \| disagreement` | `agreement \| single_sensor \| disagreement \| unknown` |
| F2 | `DecisionOutput.confidence_state` = `'low' \| 'moderate' \| 'high'` | `'low' \| 'medium' \| 'high'` |
| F3 | `Detection.quality_score`, `Detection.sun_glint_flag` are top-level | backend `Detection` forbids extras; those go under `raw_quality` |
| F4 | `FusedEvent` carries `frp_z_score`, `cluster_pixel_count`, `centroid_drift_velocity_mph`, `lulc_class`, etc. directly | backend `FusedEvent` is minimal; those are in `EventIntelligence.features` |
| F5 | Frontend `RiskBreakdown`/`ScenarioFrame` shapes are camelCase and nested differently from `EventIntelligence` | add a mapping layer in `services/api.ts` or a backend response adapter |

## Frontend logic issues

| # | Finding | Location |
|---|---|---|
| L1 | **Closed.** `generatePlumeCorridor` in `frontend/src/utils/math.ts` now uses dynamic Box-Muller Monte Carlo bearing percentile spreads (5th and 95th percentiles) matching `backend/pipeline/plume.py`, eliminating dead code. | `frontend/src/utils/math.ts` |
| L2 | **Closed.** Dual-layer proxy architecture built: Vite dev server proxies `/firms-proxy` directly to NASA, `backend/api/firms.py` proxies `/api/firms` with FastAPI CORS, and `frontend/nginx.conf` proxies `/firms-proxy/` for Docker. Open-Meteo live atmospheric wind integration added for real-time advection vectors. | `frontend/src/firms/data/firmsClient.ts`, `backend/api/firms.py`, `frontend/src/services/weather.ts` |
| L3 | **Closed.** The modal reads `GET /audit` whenever it opens, so operator actions survive a refresh. React-state audit remains only in the offline fallback (`hooks/useLocalReplay.ts`), where there is no server to persist to. | `frontend/src/DigitalTwinApp.tsx` |
| L4 | **Closed as specified.** The slider posts to `/events/{id}/simulate` (debounced) whenever the backend is up; the client-side recompute survives only as the offline fallback in `hooks/useLocalReplay.ts`. L1 still applies to that fallback path. | `frontend/src/hooks/useBackendReplay.ts` |

## ML

| # | Finding |
|---|---|
| M4 | The committed CatBoost artifact keys almost entirely on `lulc_class` / `is_in_industrial_polygon` / `distance_to_industrial_m`, saturates at ~1.0 confidence, has no trained cropland category, and mislabels the curated wildfire (→ mining) and routine-flare (→ industrial fire) scenarios. Measured values and a per-scenario comparison are in `backend/models/artifacts/README.md` §"Measured vocabulary". Needs retraining before it can carry a demo. |
| M1 | **Closed.** `backend/models/artifacts/catboost_model.cbm` is committed and wired in via `backend/pipeline/inference.py` (12 features, classes 1-5). Training code now lives alongside the artifacts (`backend/models/artifacts/train_catboost.py`, `train_isolation_forest.py`) rather than in a separate `ml/` directory. |
| M2 | **Closed for the digital twin**, which now displays `model_version` / `policy_version` straight from the event payload in the header. The FIRMS live-monitor triage panel still emits its own hard-coded strings. |
| M3 | **Partly closed.** Class probabilities now come from the trained model. Anomaly scores, segmentation masks and 90-day baselines remain hand-authored in `data/replay/*.json`; the Isolation Forest is still missing entirely, so `anomaly_model_version` stays `iforest-stub-0.1.0`. |

## Infra

| # | Finding |
|---|---|
| I1 | **Closed.** `frontend/Dockerfile` (node build → nginx) plus a `web` service in `docker-compose.yml`, published on `:8080` and gated on the API healthcheck. `docker compose up --build` now brings up the whole demo. Note the `VITE_*` values are build args, not runtime env. |
