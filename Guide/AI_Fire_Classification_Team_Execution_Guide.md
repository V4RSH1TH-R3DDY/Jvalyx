# AI Execution Guide — Space-Based Industrial Fire Classification
**Internal Team Reference | AI Input / Output / Decision Layer Only — v2: Confidence-Aware Multi-Sensor Architecture**
*(Frontend, backend, and GIS visualization are covered in separate team documents and are out of scope here.)*

---

## 1. Problem Statement

Satellite-based thermal monitoring systems (e.g., NASA FIRMS) can detect *that* a thermal anomaly is occurring, but not *what* it is. A forest fire, a controlled stubble burn, an industrial gas flare, and a refinery explosion are all currently indistinguishable on raw hotspot feeds. This system closes that gap: it fuses multi-sensor thermal data with spatial, environmental, and historical context, runs it through a confidence-aware decision pipeline, and outputs a classified, ranked hazard with an explicit confidence state and recommended action.

---

## 2. What the AI System Must Do

Given real-time thermal detections across multiple sensors, the system must:

1. Quality-check and fuse multi-sensor detections rather than treating any single feed as ground truth.
2. Classify each fused event into one of five hazard tiers (Section 3), with an explicit confidence/routing state — not a single opaque label.
3. Never let sensor disagreement collapse to "no fire" — disagreement is itself a signal that routes to verification, not suppression.
4. For high-severity or unresolved cases, delineate spatial extent and model downwind consequence as a probability corridor, not a single deterministic line.
5. Output a structured, machine-readable result — including the routing state and the reasoning behind it — that downstream systems (alerting, GIS, ops dashboards) can consume.

**Key architectural principle carried through this whole document: no single model is the sole decision-maker for Class 1/2 events.** CatBoost's classification is one input into an arbitration layer (Level 4), not the final word.

---

## 3. Classification Taxonomy

| Class | Type | Urgency Tier | Primary Risk |
|---|---|---|---|
| 1 | Accidental Industrial Fire / Explosion | Critical Emergency | Infrastructure destruction, toxic leaks, mass casualties, domino-effect explosions |
| 2 | Wildfire / Uncontrolled Forest Fire | High Emergency | Ecological destruction, rapid spread, threat to settlements |
| 3 | Uncontrolled Mining / Coal Seam Fire | Moderate Emergency | Ground subsidence, toxic subterranean gas, pit instability |
| 4 | Agricultural / Stubble Burning | Low / Seasonal | Air quality degradation, localized soil impact, compliance |
| 5 | Persistent Flare / Routine Industrial Source | Routine / Operational | Standard operational heat within permissible limits |

For Class 1 specifically: **optimize for very high recall (false negatives are far costlier than false positives), and control the resulting false-positive rate through the Level 4 arbitration logic (Section 5.6)** — sensor agreement, data-quality flags, facility-level historical FRP anomaly, temporal persistence, and the parallel anomaly detector — rather than by suppressing recall.

---

## 4. AI Task Allocation: Deterministic Pipeline vs. AI Inference

**Deterministic (non-AI) components:**
- Level 0 — per-sensor data quality/cloud/geometry checks
- Level 2 — spatiotemporal sensor fusion and agreement scoring
- Point-in-Polygon spatial verification (PostGIS / H3) against industrial boundaries
- Slope gradient (θ) from Copernicus 30m DEM
- 90-day and facility-level persistence/FRP baseline statistics
- Dual-band Planck blackbody inversion for sub-pixel combustion temperature
- Level 4 — decision engine / arbitration (deliberately kept rule-based, not a third opaque model — see Section 5.6)
- Level 6 — consequence model (population/asset overlay — see Section 5.9)
- Plume advection base physics, now wrapped in a Monte Carlo perturbation to produce a probability corridor (Section 5.8)

**AI inference components:**
- Level 3A — CatBoost multi-class triage classifier
- Level 3B — anomaly detector (Isolation Forest / One-Class SVM) running in parallel
- Level 5 — SegFormer/U-Net spatial segmentation of fire perimeter and smoke plume

---

## 5. System Architecture — Multi-Sensor Detection & Confidence-Aware Decision Pipeline

```
                        SATELLITE DATA
                             │
            ┌────────────────┼────────────────┐
            │                │                 │
         INSAT             VIIRS             MODIS
     high temporal      better spatial     complementary
     (~15min, ~4km)     (375m, 2-4x/day)   (1km, ~4x/day)
            │                │                 │
            └────────────────┼─────────────────┘
                              ↓
                   LEVEL 0 — DATA QUALITY CHECK
                              ↓
                   LEVEL 1 — FIRE EVENT DETECTOR
                              ↓
                   LEVEL 2 — SENSOR FUSION
                              ↓
            ┌─────────────────┴─────────────────┐
            ↓                                    ↓
   LEVEL 3A — CLASSIFIER               LEVEL 3B — ANOMALY MODEL
        (CatBoost)                     (Isolation Forest / OC-SVM)
            └─────────────────┬─────────────────┘
                              ↓
                   LEVEL 4 — DECISION ENGINE
                    (rule-based arbitration)
                              ↓
            ┌─────────────────┼─────────────────┐
            ↓                 ↓                  ↓
         NORMAL           UNCERTAIN            CRITICAL
            │                 │                  │
            │                 ↓                  ↓
            │       HUMAN / AI VERIFICATION   SENTINEL-2 PULL
            │            │         │              ↓
            │      confirmed   rejected    LEVEL 5 — SEGMENTATION
            │       hazard    (false alarm)        ↓
            │            │         │        LEVEL 6 — CONSEQUENCE MODEL
            │            ↓         ↓                ↓
            │      (rejoins Critical path)          │
            │            │                          │
            └────────────┴─────────────┬────────────┘
                                        ↓
                              FINAL RISK SCORE
                                        ↓
                              OPERATOR DASHBOARD
                                        ↓
                    (verified labels feed back into
                     Section 8 training data — Section 5.10)
```

### 5.1 Level 0 — Data Quality Check

Runs per-sensor, before anything is fused or classified. Each sensor's native QA/confidence fields are mapped into a unified schema:

| Sensor | Native QA fields to ingest |
|---|---|
| VIIRS | Detection confidence (nominal/low/high), cloud contamination flag, scan/track deformation, sun-glint flag |
| MODIS | QA confidence, cloud mask, sensor zenith angle |
| INSAT-3D/3DR/3DS | Cloud mask product, scan geometry, calibration status |

A detection failing quality thresholds (e.g., high cloud contamination, extreme off-nadir scan angle, sensor saturation) is flagged `low_quality` and down-weighted in Level 2 fusion rather than discarded outright — a low-quality detection is still evidence, just weaker evidence.

### 5.2 Level 1 — Fire Event Detector

The unified "tripwire": any sensor reporting a thermal anomaly above its native detection threshold generates a candidate event object (coordinates, timestamp, sensor ID, raw radiometric values, Level 0 quality flags). This replaces the earlier single-sensor (FIRMS-only) trigger with a sensor-agnostic one.

### 5.3 Level 2 — Sensor Fusion

**This is the level that needs the most explicit specification — define it precisely before building, since "sensor agreement" is meaningless without a matching rule:**

- **Spatial matching window:** sized to the coarsest sensor footprint in the comparison. A VIIRS (375m) detection is checked for INSAT (~4km) corroboration within the INSAT pixel footprint containing that coordinate, not within a fixed radius — the match tolerance should scale to sensor resolution, not be a single constant across all sensor pairs.
- **Temporal matching window:** near-simultaneous for VIIRS↔MODIS overpasses; for INSAT↔VIIRS/MODIS, match against the nearest INSAT frame(s) before and after the polar overpass, since INSAT gives a continuous trend line VIIRS/MODIS can validate against, not a single simultaneous snapshot.
- **Output — an explicit fusion state per event, not a boolean:**
  - `full_agreement` — 2+ independent sensors corroborate within the matching window
  - `temporally_confirmed_spatially_coarse` — INSAT shows a persistent signal but no VIIRS/MODIS confirmation yet (expected during the polar revisit gap — do not treat as weaker evidence, just as spatially imprecise)
  - `single_sensor_high_res` — VIIRS/MODIS detects, no INSAT trend to compare against yet
  - `disagreement` — one sensor detects, another with a clear, quality-passed view of the same coordinate at a matched time does *not* — this is the case that must route to verification, never to auto-suppression
- This produces fusion features (`sensor_agreement_state`, `sensor_count`, `spatial_match_confidence`) that feed into both Level 3 and Level 4 — fusion is a **feature generator**, not a decision-maker itself.

### 5.4 Level 3A — Classifier (CatBoost)

Unchanged from the original design (Section 6 feature matrix, `MultiClass` loss, `auto_class_weights='Balanced'`). Its output is a class + calibrated probability distribution — one input to Level 4, not a final verdict.

### 5.5 Level 3B — Anomaly Model

Isolation Forest / One-Class SVM running in parallel on the features most diagnostic of an industrial disaster (`frp_z_score`, `facility_frp_zscore`, `persistence_score`, `swir_nir_ratio`, `centroid_drift_velocity`). This exists specifically because Class 1 has only 10–20 labeled examples — nowhere near enough for CatBoost alone to learn a robust boundary, regardless of class weighting. The anomaly score is a second, independent signal into Level 4.

### 5.6 Level 4 — Decision Engine (Arbitration)

**Recommendation: keep this deterministic/rule-based (a weighted scorecard with explicit, inspectable logic), not a third ML model.** This is the safety-critical arbitration point — if a Class 1 event is ever missed, the team needs to be able to trace exactly why it was routed the way it was, in a post-incident review, without reverse-engineering another black box. A calibrated logistic combiner with monotonic constraints on interpretable inputs is an acceptable middle ground if the team wants a learned combiner later, but start rule-based.

**Inputs to Level 4:**
- CatBoost class + probability (Level 3A)
- Anomaly score (Level 3B)
- `sensor_agreement_state` (Level 2)
- Data quality flags (Level 0)
- `facility_frp_zscore` — facility-level, not just coordinate-level, anomaly (Section 6, new feature)
- `persistence_score`

**Output — one of three routing states, plus a numeric risk score:**
- **NORMAL** — high-confidence Class 3/4/5, full sensor agreement, no anomaly flag → passes straight to Final Risk Score, no human involvement.
- **UNCERTAIN** — any of: sensor disagreement, low CatBoost confidence, anomaly detector and classifier disagree with each other, or borderline facility FRP anomaly → routes to human/AI verification (Section 5.7). **This state must be able to escalate** — a confirmed hazard from verification rejoins the Critical path (segmentation + consequence model), not just get logged as "verified" and dead-end.
- **CRITICAL** — high-confidence Class 1/2, or any Class 1/2 signal regardless of confidence given the recall priority on Class 1 — triggers Sentinel-2 pull immediately (Level 5) without waiting for human sign-off, since imagery acquisition latency is the bottleneck to protect against, not a reason to gate it behind review.

### 5.7 Human / AI Verification (Uncertain branch)

Two things are bundled under "AI verification" and should be treated as distinct, cheaper-first steps before a human is involved:
1. **Automatic re-check on next pass** — cheapest option: re-query the same coordinate on the next available satellite pass (any sensor) to see if the signal persists or was transient (e.g., sun glint, a passing cloud edge misclassified). Many "uncertain" cases resolve this way without consuming human attention.
2. **Independent secondary model check** (optional, if step 1 doesn't resolve it) — a genuinely distinct check, e.g., re-running Level 3A/3B on a slightly different feature subset or a quick auxiliary check against the imagery directly, rather than just re-displaying the same CatBoost output to a human.

Only cases that survive both of the above route to an actual **human reviewer**, who sees the fused event, TreeSHAP explanation of the CatBoost output, and anomaly score. Human decision: confirm (→ rejoins Critical path) or reject (→ false alarm, logged).

### 5.8 Level 5 — Segmentation (Sentinel-2)

Unchanged in method (SegFormer/U-Net on Sentinel-2 SWIR/NIR bands), but now also triggered for verification-confirmed Uncertain cases, not only initial Critical routing.

**Plume modeling upgrade:** replace the single deterministic vector-addition trajectory with a **probability/risk corridor**. Perturb wind direction and speed within the uncertainty bounds of the GFS forecast (e.g., Monte Carlo sampling across recent GFS timesteps, or ensemble members if available) and run the advection multiple times to build a corridor at defined confidence levels (e.g., 50%/90% envelope), rather than presenting a single line as if it were exact. This is a more honest representation of what the underlying forecast actually supports.

### 5.9 Level 6 — Consequence Model (new)

Deterministic overlay, not a new learned model, consistent with the Section 4 principle of using AI only where genuine pattern recognition is required. Combines:
- Segmentation output (burn area / plume corridor, Level 5)
- Population density layer (e.g., WorldPop / GHSL — new data dependency, needs sourcing)
- Critical infrastructure proximity and criticality tags (already available via the OSM/GEM industrial asset layer)

Produces a consequence weighting that feeds into the Final Risk Score alongside the raw hazard classification — a Class 1 event over a dense populated area should score materially higher than an identical Class 1 signal in a remote area, even before human review happens.

### 5.10 Feedback Loop — Verification Into Training

Every human-confirmed or human-rejected Uncertain case should be captured as new labeled training data, not just logged for the dashboard. This is a direct, low-cost mitigation for the Class 1 data-scarcity problem (Section 8.5) — the system gets more Class 1 ground truth over time specifically from the cases it was least sure about, which is exactly where labeled data is most valuable.

---

## 6. Feature Matrix

Core 21-feature tabular vector for Level 3A/3B, plus new fusion and facility-level features required by the Level 4 architecture above.

### A. Radiometric Features (Multi-Sensor Feed)
| # | Feature | Description |
|---|---|---|
| 1 | `bright_ti4` | Brightness temperature, VIIRS 3.75 µm channel (K) |
| 2 | `bright_ti5` | Brightness temperature, VIIRS 11.0 µm channel (K) |
| 3 | `temp_ratio` | `bright_ti4 / bright_ti5` — proxy for sub-pixel combustion intensity |
| 4 | `frp` | Fire Radiative Power (MW) — instantaneous thermal energy output |
| 5 | `scan`, `track` | Pixel footprint deformation at off-nadir swath boundaries |
| 6 | `daynight` | Binary D/N flag — industrial sources burn 24/7; stubble burning drops to ~zero at night |

### B. Spatial & Topological Features (OSM & ESA WorldCover)
| # | Feature | Description |
|---|---|---|
| 7 | `is_in_industrial_polygon` | Point-in-polygon result against OSM industrial vector layers |
| 8 | `distance_to_industrial_m` | Euclidean distance to nearest industrial parcel (m) |
| 9 | `facility_type` | Categorical: `refinery`, `chemical`, `steel_mill`, `power_plant`, `none` |
| 10 | `lulc_class` | ESA WorldCover 10m class at hotspot center (Tree/Shrub/Crop/Built-up) |
| 11 | `lulc_entropy_500m` | Shannon entropy of land-use classes within 500m — separates urban from forest context |

### C. Spatio-Temporal Baseline & Kinematics (90-Day Buffer + Facility-Level)
| # | Feature | Description |
|---|---|---|
| 12 | `persistence_score` | P = detections / total overpasses, trailing 90 days, at this coordinate |
| 13 | `baseline_frp_mean` | Historical mean thermal power at this coordinate |
| 14 | `baseline_frp_std` | Historical FRP variance at this coordinate |
| 15 | `frp_z_score` | Z = (FRP_current − baseline_mean) / baseline_std, coordinate-level |
| 16 | `centroid_drift_velocity` | Fire-front movement (m/hr) vs. previous pass |
| 17 | `cluster_pixel_count` | Connected active-fire pixels within 1km |
| 17a | `facility_frp_zscore` **(new)** | Same Z-score logic, aggregated across all hotspots within a facility's OSM polygon rather than a single rounded coordinate — catches abnormal facility-wide behavior that a single-flare-point baseline would miss |

### D. Multi-Spectral Remote Sensing Features (Sentinel-2 MSI)
| # | Feature | Description |
|---|---|---|
| 18 | `swir_nir_ratio` | Band 12 (2.19 µm) / Band 8 (0.84 µm) — isolates ultra-high-temp flaring |
| 19 | `delta_nbr` | Pre- vs. post-fire Normalized Burn Ratio change — canopy damage |
| 20 | `delta_ndvi` | NDVI crash — vegetation loss magnitude |
| 21 | `thermal_plume_spread` | Bounding box area of high-SWIR anomaly (m²) |

### E. Sensor Fusion & Quality Features **(new — from Level 0 / Level 2)**
| # | Feature | Description |
|---|---|---|
| 22 | `sensor_agreement_state` | Categorical: `full_agreement`, `temporally_confirmed_spatially_coarse`, `single_sensor_high_res`, `disagreement` |
| 23 | `sensor_count` | Number of independent sensors corroborating this event |
| 24 | `data_quality_flag` | Aggregated Level 0 quality flag (cloud contamination, scan geometry, saturation) |

**Team note:** Features 1–6, 12–17, 17a, and 22–24 are cheap and always computed at trigger time. Features 7–11 require a PostGIS/vector lookup, also always computed. Features 18–21 require an on-demand Sentinel-2 pull and are needed only for Level 5 (segmentation) — don't block Level 3/4 on a Sentinel-2 fetch.

---

## 7. Live Data Ingestion Strategy

**Levels 0–4 (fusion + tripwire) feeds:**
1. Thermal triggers — polled every 15–30 min from NASA FIRMS NRT REST API (VIIRS & MODIS), and continuously from ISRO MOSDAC/Bhuvan for INSAT-3D/3DR/3DS (15-min cadence).
2. Per-sensor quality/cloud flags ingested alongside each detection (Section 5.1) — not bolted on afterward.
3. Contextual enrichment (triggered instantly on each fused event):
   - Weather: Open-Meteo API (GFS), including forecast timestep spread for plume corridor perturbation (Section 5.8)
   - Spatial/terrain: local PostGIS store (OSM vectors) + OpenTopography API (Copernicus DEM)
   - Historical baseline: internal SQL table tracking 90-day regional and facility-level FRP detections

**Level 5 (vision) feeds:**
- Queried on-demand from Copernicus Data Space Ecosystem (CDSE) OData/STAC API upon a Critical routing or a verification-confirmed Uncertain case — pulls the most recent Sentinel-2 L2A tile intersecting the hazard coordinates.

---

## 8. Training & Data Preparation

### 8.1 The Core Challenge: No Pre-Labeled Dataset

There is no existing dataset mapping the feature set to our 5-tier taxonomy. We solve this with **distant supervision** — programmatic labeling by crossing historical thermal points against geospatial boundary rules — using the NASA FIRMS Historical Archive (VIIRS 375m, India, past 2–3 years) processed with GeoPandas.

### 8.2 Automated Labeling Rules

- **Class 5 (Flares/Smelters):** Coordinate (rounded to 3 decimals) registers a thermal anomaly on >70% of days in the year → auto-label Class 5. Cross-verify against NOAA VIIRS Nightfire.
- **Class 4 (Stubble):** Coordinate intersects an ESA WorldCover Cropland polygon (Class 40) **and** falls in a peak harvest window (e.g., Oct–Nov in Punjab/Haryana) → auto-label Class 4.
- **Class 3 (Mining):** Coordinate intersects an OSM `landuse=quarry` or `mine` polygon → auto-label Class 3.
- **Class 2 (Wildfire):** Coordinate intersects an ESA WorldCover Tree Cover polygon (Class 10) **and** shows high FRP → auto-label Class 2.
- **Class 1 (Industrial Disasters):** Too rare to auto-label. Manually curate 10–20 major historical Indian industrial fires (e.g., Baghjan blowout, Vizag refinery fire), locate coordinates/dates in the FIRMS archive, and hand-label those rows. **Actively expand this set over time via the Section 5.10 verification feedback loop.**

**Class 3 vs. Class 5 tie-break rule:** A persistent coal-seam fire can satisfy the Class 5 ">70% of year" persistence rule just as easily as a genuine flare stack. **Apply rule precedence: OSM mining-polygon intersection (Class 3) overrides persistence-based Class 5 labeling.** Check `is_in_industrial_polygon` / `facility_type` mining tags *before* evaluating the persistence threshold, not after.

### 8.3 ⚠️ Known Risk — Label Circularity (must be addressed before training)

The labeling rules above use `persistence_score`, `frp_z_score`, and `lulc_class` to *generate* labels — but those same fields are also *input features*. Left unaddressed, CatBoost will partly learn to reverse-engineer our own labeling heuristic rather than the underlying phenomenon, and reported validation accuracy will be optimistic.

**Required mitigation:**
- Carve out a **held-out, human-verified test set** (target: 300–500 rows, stratified across all 5 classes) that is manually checked against satellite imagery / news reports / OSM — **not** generated purely by the auto-label rules.
- This set is used **only for final evaluation**, never for training.
- If Level 3A performance drops materially on this set vs. the auto-labeled validation split, that gap **is** the circularity effect — treat it as the real error rate.

### 8.4 Feature Matrix Construction

- Spatial lookups append slope (θ) and distance to nearest industrial polygon from local raster/vector files.
- Temporal baselines: script groups historical FIRMS data to compute 90-day FRP mean, std, and Z-score per row, plus the new facility-level aggregation (17a).
- Final output: a Pandas DataFrame with the full feature set (Section 6) + 1 label column (1–5).

### 8.5 Level 3A (CatBoost) Training Protocol

- **Loss:** Native `MultiClass` loss, 5 outputs.
- **Class imbalance handling:** `auto_class_weights='Balanced'` — reweights the loss gradient so the model is heavily penalized for missing a Class 1 event, instead of lazily predicting Class 5 for everything.
- **Categorical handling:** categorical feature indices (`facility_type`, `lulc_class`, `sensor_agreement_state`, etc.) passed directly via `cat_features`, no one-hot encoding needed.
- **Hardware:** CPU + system RAM is sufficient; 1–2M rows trains in minutes on a standard laptop, seconds with GPU.

**Class 1 data scarcity mitigation:** 10–20 manually curated Class 1 examples is not enough for CatBoost to learn a robust boundary on its own — class weighting fixes gradient emphasis, not information scarcity. This is why Level 3B (anomaly detection) runs in parallel as an independent detection path, and why the Section 5.10 feedback loop exists to grow this set over time.

### 8.6 Level 3B (Anomaly Model) Training Protocol

- Train Isolation Forest / One-Class SVM on the deterministic features most diagnostic of abnormal industrial behavior: `frp_z_score`, `facility_frp_zscore`, `persistence_score`, `swir_nir_ratio`, `centroid_drift_velocity`.
- Train primarily on Class 3/4/5 "normal" behavior to learn what routine looks like; anything scoring as an outlier within an industrial polygon is a Level 3B flag regardless of what Level 3A predicts.

### 8.7 Level 5 (SegFormer) Training Protocol

- **Base weights:** ImageNet-pretrained MiT-B0 backbone.
- **Transfer learning:** Fine-tune first on open-source EO fire-boundary datasets (Kaggle Wildfire, Omdena).
- **Custom curation target:** 200–500 manually annotated Sentinel-2 SWIR tiles (CVAT/Roboflow) covering known industrial disasters, expanded via augmentation (rotation, flip, brightness/contrast jitter). Assign an owner and deadline for this — don't leave it as background work.
- **Hardware:** MiT-B0 trains locally on a single consumer GPU (e.g., RTX 3060/4060, 6GB+ VRAM) using FP16 mixed precision and small batch sizes.

---

## 9. Validation and Testing Strategy

### 9.1 Spatial Block Cross-Validation
Random train/test splits fail on geospatial data — the model can simply memorize coordinates of known flares. Instead: divide the subcontinent into a 50km × 50km grid, hold out entire grid cells for validation.

### 9.2 Temporal Holdout
Train on 2023–2024 data, test on 2025–2026 data, to guard against seasonal shift and satellite sensor recalibration drift.

### 9.3 Evaluation Metrics
- **Macro F1-Score** — balanced precision/recall across all 5 classes.
- **Class 1 Recall (primary metric)** — target >95%.
- **Per-class precision**, tracked alongside recall — Class 5 is the overwhelming majority class; macro F1 alone can mask poor precision there, which is what drives operator alert fatigue.
- Report all metrics on **both** the auto-labeled validation split **and** the held-out human-verified set (Section 8.3).

### 9.4 Fusion & Arbitration Validation (new)
- Validate `sensor_agreement_state` thresholds against known historical events where sensor coverage is documented, to confirm the spatial/temporal matching windows (Section 5.3) aren't too strict (missing real corroboration) or too loose (falsely agreeing across unrelated events).
- Backtest Level 4's routing logic specifically on the held-out Class 1 set: confirm the arbitration rules route every known Class 1 historical event to at least Uncertain, ideally Critical — a Class 1 event routed to Normal is the single worst failure mode in this entire system and should be tested for explicitly, not just inferred from aggregate recall.

---

## 10. Known Limitations — Carry Forward Into V3 Planning

1. Label circularity between auto-labeling rules and model input features (mitigated, not eliminated, by the held-out human-verified set).
2. Class 1 training data is thin even with the anomaly-detector ensemble; the Section 5.10 feedback loop is a long-term fix, not an immediate one.
3. Level 2 fusion matching windows (spatial/temporal) need empirical tuning against real historical events (Section 9.4) before they can be trusted — the values in this doc are starting points, not final.
4. Class 3/5 boundary depends on OSM mining-tag completeness — gaps in OSM coverage for smaller/unofficial mining sites will cause misclassification toward Class 5.
5. Level 6 consequence model depends on an external population density layer (WorldPop/GHSL) not yet integrated into the ingestion pipeline — needs sourcing and licensing check.
6. Plume risk corridor quality is bounded by GFS forecast uncertainty itself — the corridor is only as good as the wind forecast spread it's built from.

---

## 11. Team Action Checklist

- [ ] Build unified Level 0 quality-flag schema across VIIRS, MODIS, and INSAT
- [ ] Implement Level 2 fusion matching logic (resolution-scaled spatial window, INSAT trend-based temporal window)
- [ ] Build 21+3 feature extraction pipeline (Section 6), split into always-computed vs. Sentinel-2-only
- [ ] Add facility-level FRP baseline (`facility_frp_zscore`) alongside existing coordinate-level baseline
- [ ] Implement auto-labeling script with the Class 3/5 precedence fix (Section 8.2)
- [ ] Assemble held-out human-verified test set — 300–500 rows, stratified, independently checked (Section 8.3)
- [ ] Train Level 3A (CatBoost) and Level 3B (Isolation Forest) as independent parallel models
- [ ] Design Level 4 as an explicit, documented rule/scorecard — write the arbitration logic down before coding it
- [ ] Define "AI verification" concretely: automatic re-pass check vs. independent secondary model (Section 5.7)
- [ ] Wire the human/AI verification outcome to (a) escalate confirmed Uncertain cases into the Critical path, and (b) feed back into training data (Section 5.10)
- [ ] Implement Monte Carlo wind perturbation for the plume probability corridor (Section 5.8)
- [ ] Source and integrate a population density layer (WorldPop/GHSL) for Level 6
- [ ] Backtest Level 4 routing specifically against the known Class 1 historical list (Section 9.4) — zero tolerance for Class 1 → Normal routing
- [ ] Set CVAT/Roboflow annotation target: 200–500 tiles for SegFormer custom set, assign owner + deadline
