import type { FireDetection } from '../types';
import type { RiskBreakdown, RouteState } from '../../types';

const BACKEND_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() || 'http://localhost:8000';

/** Result of routing one live FIRMS detection through the real backend pipeline
 * (`POST /api/triage/classify`) — the trained CatBoost classifier, the Isolation Forest
 * anomaly score, and the same arbitration/risk formulas the digital twin uses, instead of
 * `triage.ts`'s hardcoded heuristic buckets. */
export interface BackendClassification {
  classId: number;
  className: string;
  classProbabilities: Record<number, number>;
  anomalyScore: number;
  routeState: RouteState;
  confidenceState: 'low' | 'medium' | 'high';
  risk: RiskBreakdown;
  facilityFrpZscore: number;
  isInIndustrialPolygon: boolean;
  distanceToIndustrialM: number;
  matchedFacility: string | null;
  /** False whenever the model's `lulc_class` fell outside its trained vocabulary. Live
   * points are looked up against real ESA WorldCover tiles (`backend/pipeline/landcover.py`,
   * cached), so this is normally true; it goes false mainly when the live lookup fails
   * (network error, cell not yet cached) and falls back to an unclassified code, or when a
   * detection genuinely sits on a land-cover class the artifact never learned. Surface
   * this, don't hide it. */
  lulcInVocabulary: boolean;
  modelVersion: string;
  anomalyModelVersion: string;
}

/** Request body for one detection. Optional fields the backend validates strictly
 * (scan/track must be > 0, brightness_secondary >= 0) are omitted when unusable, so one
 * odd pixel can't fail a whole batch. */
function toPayload(detection: FireDetection) {
  const positive = (v: number | undefined | null) => (typeof v === 'number' && v > 0 ? v : undefined);
  const nonNegative = (v: number | undefined | null) =>
    typeof v === 'number' && Number.isFinite(v) && v >= 0 ? v : undefined;
  return {
    id: detection.id,
    latitude: detection.latitude,
    longitude: detection.longitude,
    brightness: Math.max(0, detection.brightness),
    brightness_secondary: nonNegative(detection.brightnessSecondary),
    frp: Math.max(0, detection.frp),
    scan: positive(detection.scan),
    track: positive(detection.track),
    confidence_level: detection.confidenceLevel,
    daynight: detection.daynight,
    acquired_at: detection.acquiredAt.toISOString(),
    instrument: detection.instrument,
  };
}

/** Throws on any non-2xx response or network failure — callers should fall back to the
 * local heuristic (`triageHotspot` with no `backend` argument) exactly like the digital
 * twin falls back to the offline replay pack when the backend is unreachable. */
export async function classifyDetectionBackend(
  detection: FireDetection,
  signal?: AbortSignal,
): Promise<BackendClassification> {
  const res = await fetch(`${BACKEND_BASE}/api/triage/classify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify(toPayload(detection)),
  });
  if (!res.ok) {
    throw new Error(`backend triage classify failed: ${res.status}`);
  }
  const data = await res.json();
  return {
    classId: data.class_id,
    className: data.class_name,
    classProbabilities: data.class_probabilities,
    anomalyScore: data.anomaly_score,
    routeState: data.route_state,
    confidenceState: data.confidence_state,
    risk: data.risk,
    facilityFrpZscore: data.facility_frp_zscore,
    isInIndustrialPolygon: data.is_in_industrial_polygon,
    distanceToIndustrialM: data.distance_to_industrial_m,
    matchedFacility: data.matched_facility,
    lulcInVocabulary: data.lulc_in_vocabulary,
    modelVersion: data.model_version,
    anomalyModelVersion: data.anomaly_model_version,
  };
}

export interface BatchClassification {
  results: Map<string, { classId: number; routeState: string }>;
  /** False if a chunk failed (backend down / error); `results` then holds what succeeded. */
  complete: boolean;
}

/** Classify many detections via `POST /api/triage/classify-batch`, in sequential chunks.
 * Never rejects for backend failures (returns `complete: false`); rejects only on abort. */
export async function classifyDetectionsBatch(
  detections: FireDetection[],
  options: {
    signal?: AbortSignal;
    chunkSize?: number;
    onProgress?: (done: number, total: number) => void;
  } = {},
): Promise<BatchClassification> {
  const { signal, chunkSize = 1000, onProgress } = options;
  const results: BatchClassification['results'] = new Map();
  for (let start = 0; start < detections.length; start += chunkSize) {
    const chunk = detections.slice(start, start + chunkSize);
    let data: { results: { id: string; class_id: number; route_state: string }[] };
    try {
      const res = await fetch(`${BACKEND_BASE}/api/triage/classify-batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal,
        body: JSON.stringify({ detections: chunk.map(toPayload) }),
      });
      if (!res.ok) throw new Error(`backend batch classify failed: ${res.status}`);
      data = await res.json();
    } catch (err) {
      if (signal?.aborted) throw err;
      console.warn('[triage] batch classification failed, falling back to heuristic:', err);
      return { results, complete: false };
    }
    for (const r of data.results) results.set(r.id, { classId: r.class_id, routeState: r.route_state });
    onProgress?.(Math.min(start + chunk.length, detections.length), detections.length);
  }
  return { results, complete: true };
}
