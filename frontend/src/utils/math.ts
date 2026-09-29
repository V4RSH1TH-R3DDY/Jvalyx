import type { RiskBreakdown, RouteState, SensorAgreementState, PlumeEnvelope } from '../types';

export const EARTH_RADIUS_M = 6_371_000.0;

/**
 * Great-circle distance between two coordinates in metres
 */
export function haversine_m(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const rLat1 = toRad(lat1);
  const rLat2 = toRad(lat2);

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(rLat1) * Math.cos(rLat2) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return EARTH_RADIUS_M * c;
}

/**
 * Temperature contrast ratio between SWIR (ti4) and TIR (ti5)
 */
export function temperature_ratio(ti4_k: number, ti5_k: number, eps: number = 1e-6): number {
  return Number((ti4_k / Math.max(ti5_k, eps)).toFixed(3));
}

/**
 * Sigmoid activation function
 */
export function sigmoid(x: number): number {
  return 1.0 / (1.0 + Math.exp(-x));
}

/**
 * Transparent Risk Score Calculation from Section 12.4
 * R = 100 * clip(0.35 * Severity + 0.25 * Anomaly + 0.20 * Spread + 0.20 * Exposure, 0, 1)
 */
export function computeRiskScore(
  classProbs: Record<number, number>,
  facilityZ: number,
  clusterPixelCount: number,
  driftMph: number,
  exposure: number
): RiskBreakdown {
  const classWeights: Record<number, number> = {
    1: 1.00, // Unusual Industrial Fire
    2: 0.80, // Wildfire
    3: 0.55, // Mining / Coal-seam
    4: 0.20, // Agriculture
    5: 0.05  // Routine Industrial Heat / Flare
  };

  const severity = Object.entries(classProbs).reduce((acc, [cls, p]) => {
    return acc + (classWeights[Number(cls)] || 0) * p;
  }, 0);

  const anomaly = sigmoid((facilityZ - 3.0) / 1.0);
  const spread = sigmoid(0.45 * Math.log1p(clusterPixelCount) + 0.002 * driftMph - 1.2);
  const expScore = Math.max(0, Math.min(1, exposure));

  const raw = 0.35 * severity + 0.25 * anomaly + 0.20 * spread + 0.20 * expScore;
  const total = Math.round(100 * Math.max(0, Math.min(1, raw)));

  return {
    total,
    severity: Number(severity.toFixed(3)),
    anomaly: Number(anomaly.toFixed(3)),
    spread: Number(spread.toFixed(3)),
    exposure: Number(expScore.toFixed(3))
  };
}

/**
 * Deterministic arbitration engine from Section 13
 */
export interface ArbitrationConfig {
  class1_threshold: number;
  class2_threshold: number;
  facility_z_threshold: number;
  anomaly_threshold: number;
  min_quality: number;
  min_model_confidence: number;
}

export const DEFAULT_ARBITRATION_CONFIG: ArbitrationConfig = {
  class1_threshold: 0.60,
  class2_threshold: 0.55,
  facility_z_threshold: 4.0,
  anomaly_threshold: 0.75,
  min_quality: 0.45,
  min_model_confidence: 0.55
};

/**
 * Port of `backend/pipeline/arbitration.py::arbitrate`. Keep the two in sync — this is
 * only exercised by the offline (no-backend) fallback, so drift here is invisible until
 * someone is actually running disconnected.
 */
export function routeEvent(
  probs: Record<number, number>,
  anomalyScore: number,
  isIndustrial: boolean,
  facilityZ: number,
  fusionState: SensorAgreementState,
  qualityScore: number,
  cfg: ArbitrationConfig = DEFAULT_ARBITRATION_CONFIG,
  lulcInVocabulary: boolean = true,
  historyComplete: boolean = true,
  onWater: boolean = false
): RouteState {
  const p1 = probs[1] || 0;
  const p2 = probs[2] || 0;
  const entries = Object.entries(probs);
  const maxP = entries.length ? Math.max(...entries.map(([, p]) => p)) : 0;

  const disagreement = fusionState === 'disagreement';

  const strongModelCritical = (p1 >= cfg.class1_threshold || p2 >= cfg.class2_threshold) && !disagreement;
  const industrialAnomalyCritical =
    isIndustrial &&
    facilityZ >= cfg.facility_z_threshold &&
    qualityScore >= cfg.min_quality &&
    !disagreement;

  const critical = strongModelCritical || industrialAnomalyCritical;

  const uncertain =
    disagreement ||
    anomalyScore >= cfg.anomaly_threshold ||
    qualityScore < cfg.min_quality ||
    maxP < cfg.min_model_confidence ||
    !lulcInVocabulary ||
    !historyComplete;

  if (disagreement || !historyComplete) return 'UNCERTAIN';
  // Water land cover under an "unusual industrial fire" call is mostly ash ponds,
  // reservoirs and river banks next to routine sites — never auto-escalate it.
  const topClass = entries.length
    ? Number(entries.reduce((a, b) => (b[1] > a[1] ? b : a))[0])
    : null;
  if (onWater && topClass === 1) return 'UNCERTAIN';
  if (critical) return 'CRITICAL';
  if (uncertain) return 'UNCERTAIN';
  return 'NORMAL';
}

/**
 * Monte Carlo Downwind Plume Probability Corridor
 * Generates 50% and 90% uncertainty dispersion envelopes
 */
export function generatePlumeCorridor(
  origin: [number, number], // [lat, lon]
  windSpeedMps: number,
  windDirectionDeg: number, // direction TOWARD which wind travels
  samples: number = 250
): PlumeEnvelope {
  const [lat0, lon0] = origin;
  // Convert wind speed to km/h projection
  const baseDistKm = Math.max(1.5, windSpeedMps * 0.4);

  const endpoints: [number, number][] = [];
  // Generate Monte Carlo samples
  for (let i = 0; i < samples; i++) {
    // Normal distribution approximation via Box-Muller
    const u1 = Math.random() || 0.001;
    const u2 = Math.random() || 0.001;
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);

    const sampledSpeed = Math.max(0.5, windSpeedMps + z0 * 1.5);
    const sampledDir = (windDirectionDeg + z1 * 12.0) % 360;

    const distKm = Math.max(1.0, sampledSpeed * 0.4);
    const rad = (sampledDir * Math.PI) / 180;

    // Flat earth km to degree conversion
    const dLat = (distKm * Math.cos(rad)) / 111.0;
    const dLon = (distKm * Math.sin(rad)) / (111.0 * Math.cos((lat0 * Math.PI) / 180));
    endpoints.push([lat0 + dLat, lon0 + dLon]);
  }

  function pct(values: number[], q: number): number {
    const ordered = [...values].sort((a, b) => a - b);
    return ordered[Math.min(ordered.length - 1, Math.floor(q * ordered.length))];
  }

  // Angular half-width of the corridor derived from the sampled bearing spread
  const bearings = endpoints.map(([lat, lon]) => {
    const angle = (Math.atan2(lon - lon0, lat - lat0) * 180) / Math.PI;
    return (((angle - windDirectionDeg + 180) % 360) + 360) % 360 - 180;
  });

  const half90Deg = Math.max(12.0, (pct(bearings, 0.95) - pct(bearings, 0.05)) / 2);
  const half50Deg = Math.max(6.0, half90Deg * 0.5);

  const spread90Rad = (half90Deg * Math.PI) / 180;
  const spread50Rad = (half50Deg * Math.PI) / 180;

  const radCentral = (windDirectionDeg * Math.PI) / 180;
  const centralLat = lat0 + (baseDistKm * Math.cos(radCentral)) / 111.0;
  const centralLon = lon0 + (baseDistKm * Math.sin(radCentral)) / (111.0 * Math.cos((lat0 * Math.PI) / 180));

  const left90Lat = lat0 + (baseDistKm * 1.15 * Math.cos(radCentral - spread90Rad)) / 111.0;
  const left90Lon = lon0 + (baseDistKm * 1.15 * Math.sin(radCentral - spread90Rad)) / (111.0 * Math.cos((lat0 * Math.PI) / 180));

  const right90Lat = lat0 + (baseDistKm * 1.15 * Math.cos(radCentral + spread90Rad)) / 111.0;
  const right90Lon = lon0 + (baseDistKm * 1.15 * Math.sin(radCentral + spread90Rad)) / (111.0 * Math.cos((lat0 * Math.PI) / 180));

  const left50Lat = lat0 + (baseDistKm * 1.05 * Math.cos(radCentral - spread50Rad)) / 111.0;
  const left50Lon = lon0 + (baseDistKm * 1.05 * Math.sin(radCentral - spread50Rad)) / (111.0 * Math.cos((lat0 * Math.PI) / 180));

  const right50Lat = lat0 + (baseDistKm * 1.05 * Math.cos(radCentral + spread50Rad)) / 111.0;
  const right50Lon = lon0 + (baseDistKm * 1.05 * Math.sin(radCentral + spread50Rad)) / (111.0 * Math.cos((lat0 * Math.PI) / 180));

  return {
    centerline: [
      [lat0, lon0],
      [centralLat, centralLon]
    ],
    cone90: [
      [lat0, lon0],
      [left90Lat, left90Lon],
      [centralLat, centralLon],
      [right90Lat, right90Lon],
      [lat0, lon0]
    ],
    cone50: [
      [lat0, lon0],
      [left50Lat, left50Lon],
      [centralLat, centralLon],
      [right50Lat, right50Lon],
      [lat0, lon0]
    ],
    windSpeedMps,
    windDirectionDeg
  };
}
