# **AI Execution Guide — Space-Based Industrial Fire Classification**

**Internal Team Reference | AI Input / Output / Decision Layer Only** *(Frontend, backend, and GIS visualization should be  covered in separate team documents and are out of scope here.)*

---

## **1\. Problem Statement**

Satellite-based thermal monitoring systems (e.g., NASA FIRMS) can detect *that* a thermal anomaly is occurring, but not *what* it is. A forest fire, a controlled stubble burn, an industrial gas flare, and a refinery explosion are all currently indistinguishable on raw hotspot feeds. This system closes that gap: it ingests thermal, spatial, environmental, and historical data, and outputs a classified, ranked hazard with an associated confidence and recommended action.

---

## **2\. What the AI System Must Do**

Given a real-time thermal detection, the system must:

1. Enrich the raw hotspot with spatial, historical, and meteorological context.  
2. Classify it into one of five hazard tiers (Section 3).  
3. For high-severity classes, delineate the precise spatial extent of the fire/plume.  
4. Output a structured, machine-readable result that downstream systems (alerting, GIS, ops dashboards — handled by other team workstreams) can consume.

---

## **3\. Classification Taxonomy**

| Class | Type | Urgency Tier | Primary Risk |
| ----- | ----- | ----- | ----- |
| 1 | Accidental Industrial Fire / Explosion | Critical Emergency | Infrastructure destruction, toxic leaks, mass casualties, domino-effect explosions |
| 2 | Wildfire / Uncontrolled Forest Fire | High Emergency | Ecological destruction, rapid spread, threat to settlements |
| 3 | Uncontrolled Mining / Coal Seam Fire | Moderate Emergency | Ground subsidence, toxic subterranean gas, pit instability |
| 4 | Agricultural / Stubble Burning | Low / Seasonal | Air quality degradation, localized soil impact, compliance |
| 5 | Persistent Flare / Routine Industrial Source | Routine / Operational | Standard operational heat within permissible limits |

**Classes 1 and 2 trigger immediate downstream escalation** (see Section 9, Decision Logic).

---

## **4\. AI Task Allocation: Deterministic Pipeline vs. AI Inference**

To keep latency low and reliability high, physics/geometry problems are **not** handed to a model. Only genuine pattern-recognition problems go to AI.

**Deterministic (non-AI) pipeline:**

* Point-in-Polygon spatial verification (PostGIS / H3) against industrial boundaries  
* Slope gradient (θ) from Copernicus 30m DEM  
* 90-day persistence ratio, historical FRP mean/std, and Z-score computation  
* Dual-band Planck blackbody inversion for sub-pixel combustion temperature  
* Plume advection via vector addition of real-time GFS wind vectors *(see Section 10 — this is a first-order simplification; flag as a known limitation, not a bug)*

**AI inference pipeline:**

* Multi-modal triage classification: mapping the 21-feature tabular vector to one of 5 hazard classes  
* Multi-spectral spatial segmentation: fire perimeter and smoke plume delineation from imagery

---

## **5\. Model Architecture**

### **Stage 1 — Primary Triage Classifier**

* **Model:** CatBoost (Categorical Boosting)  
* **Task:** Continuous real-time classification of every incoming thermal trigger into Classes 1–5  
* **Input:** 21-dimensional tabular vector (mixed numeric \+ categorical) — full spec in Section 6  
* **Why CatBoost:** Native handling of high-cardinality categoricals (land-cover IDs, OSM tags) without one-hot blowup; supports custom cost matrices to penalize Class 1 false negatives; TreeSHAP explainability for post-hoc audit of any alert  
* **Rejected alternatives:** TabNet (overfitting risk, opacity, not worth it at this data scale); Random Forest (struggles with the \~10,000:1 class imbalance and gives poorly calibrated probability outputs)  
* **Source:** Yandex Open-Source (`catboost/catboost`)

### **Stage 2 — Tactical Vision Segmenter**

* **Model:** SegFormer (MiT-B0 backbone) or lightweight U-Net  
* **Task:** Triggered **only** on Class 1/2 alerts — maps exact burn/smoke pixel boundaries at 20m resolution  
* **Input:** Sentinel-2 multi-spectral cutout (SWIR 12, SWIR 11, NIR 8, Red 4\) as an H×W×C tensor  
* **Why segmentation over detection/classification:** YOLO-style bounding boxes swallow large non-burning background and can't trace irregular fire lines; ResNet/ViT-style whole-tile classification can't localize which specific asset within an industrial complex is burning. A pixel mask is what tactical response actually needs.  
* **Rejected alternatives:** YOLO/RT-DETR, ResNet/ViT classification  
* **Source:** Hugging Face Transformers (SegFormer) or SMP (`qubvel/segmentation_models.pytorch`)

**Why two stages, not one:** Running Stage 2 continuously across the whole subcontinent is computationally infeasible in real time. Stage 1 is the cheap, always-on "tripwire"; it only calls Stage 2 when the stakes justify the compute.

---

## **6\. Feature Matrix — Full 21-Feature Specification**

This is the exact input vector Stage 1 (CatBoost) consumes. Every row should map directly to a column in the training DataFrame.

### **A. Radiometric Features (NASA FIRMS Feed)**

| \# | Feature | Description |
| ----- | ----- | ----- |
| 1 | `bright_ti4` | Brightness temperature, VIIRS 3.75 µm channel (K) |
| 2 | `bright_ti5` | Brightness temperature, VIIRS 11.0 µm channel (K) |
| 3 | `temp_ratio` | `bright_ti4 / bright_ti5` — proxy for sub-pixel combustion intensity |
| 4 | `frp` | Fire Radiative Power (MW) — instantaneous thermal energy output |
| 5 | `scan`, `track` | Pixel footprint deformation at off-nadir swath boundaries |
| 6 | `daynight` | Binary D/N flag — industrial sources burn 24/7; stubble burning drops to \~zero at night |

### **B. Spatial & Topological Features (OSM & ESA WorldCover)**

| \# | Feature | Description |
| ----- | ----- | ----- |
| 7 | `is_in_industrial_polygon` | Point-in-polygon result against OSM industrial vector layers |
| 8 | `distance_to_industrial_m` | Euclidean distance to nearest industrial parcel (m) |
| 9 | `facility_type` | Categorical: `refinery`, `chemical`, `steel_mill`, `power_plant`, `none` |
| 10 | `lulc_class` | ESA WorldCover 10m class at hotspot center (Tree/Shrub/Crop/Built-up) |
| 11 | `lulc_entropy_500m` | Shannon entropy of land-use classes within 500m — separates urban from forest context |

### **C. Spatio-Temporal Baseline & Kinematics (90-Day Buffer)**

| \# | Feature | Description |
| ----- | ----- | ----- |
| 12 | `persistence_score` | P \= detections / total overpasses, trailing 90 days |
| 13 | `baseline_frp_mean` | Historical mean thermal power at this coordinate |
| 14 | `baseline_frp_std` | Historical FRP variance at this coordinate |
| 15 | `frp_z_score` | Z \= (FRP\_current − baseline\_mean) / baseline\_std |
| 16 | `centroid_drift_velocity` | Fire-front movement (m/hr) vs. previous pass |
| 17 | `cluster_pixel_count` | Connected active-fire pixels within 1km |

### **D. Multi-Spectral Remote Sensing Features (Sentinel-2 MSI)**

| \# | Feature | Description |
| ----- | ----- | ----- |
| 18 | `swir_nir_ratio` | Band 12 (2.19 µm) / Band 8 (0.84 µm) — isolates ultra-high-temp flaring |
| 19 | `delta_nbr` | Pre- vs. post-fire Normalized Burn Ratio change — canopy damage |
| 20 | `delta_ndvi` | NDVI crash — vegetation loss magnitude |
| 21 | `thermal_plume_spread` | Bounding box area of high-SWIR anomaly (m²) |

**Team note:** Features 1–6 and 12–17 are available at trigger time from FIRMS \+ the internal SQL baseline store (cheap, always computed). Features 7–11 require a PostGIS/vector lookup (cheap, always computed). Features 18–21 require an on-demand Sentinel-2 pull and are **only needed for Stage 2**, not Stage 1 — don't block the Stage 1 tripwire on a Sentinel-2 fetch.

---

## **7\. Live Data Ingestion Strategy**

**Model 1 (tripwire) feeds:**

1. Thermal trigger — polled every 15–30 min from NASA FIRMS NRT REST API (VIIRS & MODIS)  
2. Contextual enrichment (triggered instantly on each new hotspot):  
   * Weather: Open-Meteo API (GFS)  
   * Spatial/terrain: local PostGIS store (OSM vectors) \+ OpenTopography API (Copernicus DEM)  
   * Historical baseline: internal SQL table tracking 90-day regional FIRMS detections

**Model 2 (vision) feeds:**

* Queried on-demand from Copernicus Data Space Ecosystem (CDSE) OData/STAC API **only** upon a Class 1 or 2 trigger — pulls the most recent Sentinel-2 L2A tile intersecting the hazard coordinates.

---

## **8\. Training & Data Preparation**

### **8.1 The Core Challenge: No Pre-Labeled Dataset**

There is no existing dataset mapping the 21 features to our 5-tier taxonomy. FIRMS gives raw coordinates, not labels. We solve this with **distant supervision** — programmatic labeling by crossing historical thermal points against geospatial boundary rules — using the NASA FIRMS Historical Archive (VIIRS 375m, India, past 2–3 years) processed with GeoPandas.

### **8.2 Automated Labeling Rules**

* **Class 5 (Flares/Smelters):** Coordinate (rounded to 3 decimals) registers a thermal anomaly on \>70% of days in the year → auto-label Class 5\. Cross-verify against NOAA VIIRS Nightfire.  
* **Class 4 (Stubble):** Coordinate intersects an ESA WorldCover Cropland polygon (Class 40\) **and** falls in a peak harvest window (e.g., Oct–Nov in Punjab/Haryana) → auto-label Class 4\.  
* **Class 3 (Mining):** Coordinate intersects an OSM `landuse=quarry` or `mine` polygon → auto-label Class 3\.  
* **Class 2 (Wildfire):** Coordinate intersects an ESA WorldCover Tree Cover polygon (Class 10\) **and** shows high FRP → auto-label Class 2\.  
* **Class 1 (Industrial Disasters):** Too rare to auto-label. Manually curate 10–20 major historical Indian industrial fires (e.g., Baghjan blowout, Vizag refinery fire), locate coordinates/dates in the FIRMS archive, and hand-label those rows.

**Refinement — Class 3 vs. Class 5 tie-break rule:** A persistent coal-seam fire can satisfy the Class 5 "\>70% of year" persistence rule just as easily as a genuine flare stack. **Apply rule precedence: OSM mining-polygon intersection (Class 3\) overrides persistence-based Class 5 labeling.** Check `is_in_industrial_polygon` / `facility_type` mining tags *before* evaluating the persistence threshold, not after.

### **8.3 ⚠️ Known Risk — Label Circularity (must be addressed before training)**

The labeling rules above use `persistence_score`, `frp_z_score`, and `lulc_class` to *generate* labels — but those same fields are also *input features* (7, 10, 12, 15 in Section 6). Left unaddressed, CatBoost will partly learn to reverse-engineer our own labeling heuristic rather than the underlying phenomenon, and reported validation accuracy will be optimistic.

**Required mitigation:**

* Carve out a **held-out, human-verified test set** (target: 300–500 rows, stratified across all 5 classes) that is manually checked against satellite imagery / news reports / OSM — **not** generated purely by the auto-label rules.  
* This set is used **only for final evaluation**, never for training. It's the number you actually trust and report internally.  
* If Stage-1 performance drops materially on this set vs. the auto-labeled validation split, that gap **is** the circularity effect — treat it as the real error rate.

### **8.4 Feature Matrix Construction**

* Spatial lookups append slope (θ) and distance to nearest industrial polygon from local raster/vector files.  
* Temporal baselines: script groups historical FIRMS data to compute 90-day FRP mean, std, and Z-score per row.  
* Final output: a Pandas DataFrame with 21 feature columns (numeric \+ categorical) \+ 1 label column (1–5).

### **8.5 Model 1 (CatBoost) Training Protocol**

* **Loss:** Native `MultiClass` loss, 5 outputs.  
* **Class imbalance handling:** `auto_class_weights='Balanced'` — reweights the loss gradient so the model is heavily penalized for missing a Class 1 event, instead of lazily predicting Class 5 for everything.  
* **Categorical handling:** categorical feature indices (facility\_type, lulc\_class, etc.) passed directly via `cat_features`, no one-hot encoding needed.  
* **Hardware:** CPU \+ system RAM is sufficient; 1–2M rows trains in minutes on a standard laptop, seconds with GPU.

**Refinement — Class 1 data scarcity mitigation:** 10–20 manually curated Class 1 examples is not enough for CatBoost to learn a robust boundary on its own, no matter how the loss is weighted — class weighting fixes *gradient emphasis*, not *information scarcity*. Run an **anomaly-detection model in parallel** (Isolation Forest or One-Class SVM) on the deterministic features most diagnostic of an industrial disaster (`frp_z_score`, `persistence_score`, `swir_nir_ratio`, `centroid_drift_velocity`). Flag as Class 1 candidate if **either** CatBoost predicts Class 1 **or** the anomaly detector flags an outlier inside an industrial polygon. This gives a second, independent detection path for the highest-stakes, lowest-data class instead of relying on one small-sample classifier alone.

### **8.6 Model 2 (SegFormer) Training Protocol**

* **Base weights:** ImageNet-pretrained MiT-B0 backbone.  
* **Transfer learning:** Fine-tune first on open-source EO fire-boundary datasets (Kaggle Wildfire, Omdena).  
* **Custom curation:** Sentinel-2 SWIR cutouts of known industrial disasters (from the Class 1 manual list), annotated in CVAT or Roboflow for precise ground-truth masks.

**Refinement — quantify the custom annotation target:** Don't leave this open-ended. Target **200–500 manually annotated tiles** as the custom industrial-plume set, expanded via augmentation (rotation, flip, brightness/contrast jitter) to increase effective training volume without requiring proportionally more manual annotation hours. Track this as a concrete team task with an owner and a deadline, not a background activity.

* **Hardware:** MiT-B0 is light enough to train locally on a single consumer GPU (e.g., RTX 3060/4060, 6GB+ VRAM) using FP16 mixed precision and small batch sizes.

---

## **9\. Validation and Testing Strategy**

### **9.1 Spatial Block Cross-Validation**

Random train/test splits fail on geospatial data — the model can simply memorize coordinates of known flares. Instead: divide the subcontinent into a 50km × 50km grid, hold out entire grid cells for validation. This forces the model to classify unseen geography using only the engineered features (slope, Z-score, LULC), not memorized location.

### **9.2 Temporal Holdout**

Train on 2023–2024 data, test on 2025–2026 data, to guard against seasonal shift and satellite sensor recalibration drift over time.

### **9.3 Evaluation Metrics**

Accuracy is discarded — it's meaningless under this class imbalance. Use instead:

* **Macro F1-Score** — balanced precision/recall across all 5 classes  
* **Class 1 Recall (primary metric)** — target \>95%, accepting more false positives (extra flare alerts) to guarantee we don't miss a real refinery explosion  
* **Refinement — per-class precision, tracked alongside recall:** Class 5 is the overwhelming majority class. Macro F1 alone can mask poor precision on it. Track per-class precision explicitly — low Class 5 precision (i.e., lots of routine flares getting misclassified upward) is what drives operator alert fatigue and erodes trust in the system over time, even if Class 1 recall looks great.  
* Report all metrics on **both** the auto-labeled validation split **and** the held-out human-verified set (Section 8.3) — the gap between the two is your real circularity-adjusted performance.

---

## **10\. Decision Logic — From Model Output to Action**

*(This is in scope as "decision-making" per the AI layer's responsibility — it stops at handing off a structured decision object, not at how it's displayed.)*

* **Class 1 or 2 output → immediate Stage 2 trigger** (imagery pull \+ segmentation), regardless of confidence score, given the cost asymmetry favors over-triggering.  
* **Confidence thresholding:** attach CatBoost's class probability to every output. Recommend a two-tier response:  
  * High confidence Class 1/2 → auto-escalate to emergency notification pipeline (owned by ops/backend team)  
  * Lower confidence Class 1/2, or an Isolation Forest flag without CatBoost agreement → route to a **human verification queue** rather than auto-escalating, given false alarms on true emergencies carry real credibility cost  
* **TreeSHAP output should accompany every Class 1/2 alert** handed downstream — whoever reviews the alert (human or automated) should see *why* the model flagged it (which features drove the decision), not just the class label.  
* Known limitation to flag to the team: plume advection is currently simple vector addition of GFS wind, not a proper Gaussian dispersion model. This is an acceptable MVP simplification but should not be presented as physically precise — note it as a v2 item if toxic dispersion trajectories become a load-bearing part of the alert.

---

## **11\. Known Limitations — Carry Forward Into V2 Planning**

1. Label circularity between auto-labeling rules and model input features (mitigated, not eliminated, by the held-out human-verified set).  
2. Class 1 training data is extremely thin (10–20 examples) — anomaly-detection ensemble is a mitigation, not a full fix. Actively look for more historical Class 1 incidents to expand this set over time.  
3. Plume advection model is first-order (vector addition), not a full Gaussian plume dispersion model.  
4. Class 3/5 boundary depends on OSM mining-tag completeness — gaps in OSM coverage for smaller/unofficial mining sites will cause misclassification toward Class 5\.

---

## **13\. Data Sourcing & API Endpoints**

A crucial component of this architecture is securing the underlying remote-sensing feeds. This system relies entirely on free, open-access Earth Observation (EO) data.

To prevent bottlenecks, the data sourcing strategy is divided into two distinct modes: **Historical/Bulk Download** (for generating the training datasets) and **Real-Time API Interrogation** (for the operational deployment).

### 13.1 Historical Sourcing (For Model Training)

Before the models can be trained, you must construct the historical feature matrix. This requires downloading bulk historical data archives.

* **NASA FIRMS Active Fire Data (Historical CSVs):**  
  * **Source:** NASA Earthdata / FIRMS Archive  
  * **Access Method:** You will use the FIRMS Data Download portal to request historical VIIRS (375m) and MODIS (1km) data.  
  * **Format:** CSV files.  
  * **Required Action:** Download the specific bounding box for the Indian subcontinent covering the past 2–3 years. This will serve as the base dataset for your distant supervision labeling script.  
* **Copernicus Sentinel-2 Imagery (For SegFormer Training):**  
  * **Source:** Copernicus Data Space Ecosystem (CDSE)  
  * **Access Method:** You will query the CDSE catalogue using their STAC (SpatioTemporal Asset Catalog) API or OData REST protocols to locate specific historical tiles.  
    Remote Sensing Datasets & Sensors Comparison  
  * **Format:** Cloud-Optimized GeoTIFFs (Bands 12, 11, 8, 4).  
  * **Required Action:** Use the manual Class 1 curated list to query the STAC API for historical imagery of those specific disaster dates.

### **13.2 Real-Time API Endpoints (For Operational Deployment)**

Once the models are trained and deployed, the backend must continuously poll live feeds to generate the 21-feature input array.

* **The Tripwire Trigger (NASA FIRMS NRT API):**  
  * **Endpoint / Source:** `[https://firms.modaps.eosdis.nasa.gov/api/area/csv/](https://firms.modaps.eosdis.nasa.gov/api/area/csv/)`  
  * **Description:** This REST API provides Near Real-Time (NRT) active fire detections.  
  * **Usage:** Your backend will query this endpoint every 15–30 minutes, fetching the latest CSV output for your designated bounding box to trigger the Stage 1 CatBoost triage.  
* **Tactical Vision Pipeline (Copernicus CDSE STAC API):**  
  * **Endpoint / Source:** Copernicus Data Space Ecosystem (CDSE) STAC API (`[https://browser.stac.dataspace.copernicus.eu/](https://browser.stac.dataspace.copernicus.eu/)`)  
  * **Description:** The STAC API is a standardized interface designed by the Earth Observation community to streamline data discovery and retrieval.  
  * **Usage:** When Stage 1 flags a Class 1 or 2 event, the backend will dynamically query this STAC API to find and download the most recent Sentinel-2 Level-2A surface reflectance product intersecting the hazard coordinates.  
* **Meteorological Advection Data (Open-Meteo):**  
  * **Endpoint / Source:** `[https://api.open-meteo.com/v1/forecast](https://api.open-meteo.com/v1/forecast)`  
  * **Description:** A free, open-source weather API that aggregates NOAA GFS and ECMWF models.  
  * **Usage:** Used to fetch real-time pressure-level wind vectors for the downstream plume advection modeling.

