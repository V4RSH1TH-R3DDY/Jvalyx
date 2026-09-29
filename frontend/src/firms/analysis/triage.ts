import type { FireDetection } from '../types';
import { PROTECTED_AREAS } from '../config/protectedAreas';
import { KNOWN_FACILITIES } from '../../services/firms';
import {
  computeRiskScore,
  generatePlumeCorridor,
  haversine_m,
  routeEvent,
} from '../../utils/math';
import type { RiskBreakdown, RouteState } from '../../types';
import type { LiveWeatherData } from '../../services/weather';
import { matchIndustrialPolygon, distanceToNearestIndustrialM } from './industrialSpatial';
import type { BackendClassification } from './backendClassify';

export interface HotspotTriage {
  detection: FireDetection;
  context: {
    nearestFacility: { name: string; distanceM: number } | null;
    industrialPolygon: {
      name: string;
      zone: string;
      areaSqkm?: number;
      isMine: boolean;
    } | null;
    inProtectedArea: { name: string; category: string } | null;
    isIndustrial: boolean;
    facilityZ: number;
    biome: 'industrial' | 'protected-forest' | 'open-terrain';
    weather?: LiveWeatherData;
  };
  classProbabilities: Record<number, number>;
  classId: number;
  className: string;
  anomalyScore: number;
  routeState: RouteState;
  risk: RiskBreakdown;
  plume: ReturnType<typeof generatePlumeCorridor>;
  explanation: string[];
  disclaimer: string;
  /** 'model' when classified by the trained backend (`/api/triage/classify`); 'heuristic'
   * when it fell back to this file's local bucket table (backend unreachable/offline). */
  source: 'model' | 'heuristic';
  /** False when the backend's land-cover feature had no real signal for this point
   * (no live LULC source is wired in yet). Always true for 'heuristic' (n/a). */
  lulcInVocabulary: boolean;
  modelVersion?: string;
}

const CLASS_NAMES: Record<number, string> = {
  1: 'Unusual Industrial Fire',
  2: 'Wildfire or Forest Fire',
  3: 'Uncontrolled Mining / Coal-Seam Fire',
  4: 'Agricultural / Stubble Burning',
  5: 'Routine Industrial Heat / Flare',
};

const STUBBLE_MONTHS = new Set([9, 10, 3, 4]); // Sep-Oct (kharif), Mar-Apr (rabi)

export function triageHotspot(
  detection: FireDetection,
  weather?: LiveWeatherData,
  backend?: BackendClassification,
): HotspotTriage {
  // 1. Check direct point-in-polygon containment against 278 curated industrial & mining polygons
  const polyMatch = matchIndustrialPolygon(detection.latitude, detection.longitude);
  const polyDist = distanceToNearestIndustrialM(detection.latitude, detection.longitude);

  let industrialPolygon: HotspotTriage['context']['industrialPolygon'] = null;
  let nearestFacility: { name: string; distanceM: number } | null = null;
  let facilityBaselineMean = 12;
  let facilityBaselineStd = 6;

  if (polyMatch) {
    const isMine = Boolean(
      (polyMatch.coal_industrial_zone && polyMatch.coal_industrial_zone.toLowerCase().includes('coal')) ||
      polyMatch.industrial === 'mine' ||
      (polyMatch.name && polyMatch.name.toLowerCase().includes('coal'))
    );
    industrialPolygon = {
      name: polyMatch.name || polyMatch.coal_industrial_zone?.replace(/_/g, ' ') || 'Industrial Complex',
      zone: polyMatch.coal_industrial_zone ? polyMatch.coal_industrial_zone.replace(/_/g, ' ') : 'General Industrial',
      areaSqkm: polyMatch.area_sqkm,
      isMine,
    };
    nearestFacility = { name: industrialPolygon.name, distanceM: 0 };
    facilityBaselineMean = isMine ? 22 : 38.4;
    facilityBaselineStd = isMine ? 6 : 8.0;
  } else if (polyDist.match && polyDist.distanceM <= 3500) {
    const isMine = Boolean(
      (polyDist.match.coal_industrial_zone && polyDist.match.coal_industrial_zone.toLowerCase().includes('coal')) ||
      polyDist.match.industrial === 'mine'
    );
    industrialPolygon = {
      name: polyDist.match.name || polyDist.match.coal_industrial_zone?.replace(/_/g, ' ') || 'Industrial Complex',
      zone: polyDist.match.coal_industrial_zone ? polyDist.match.coal_industrial_zone.replace(/_/g, ' ') : 'General Industrial',
      areaSqkm: polyDist.match.area_sqkm,
      isMine,
    };
    nearestFacility = { name: industrialPolygon.name, distanceM: polyDist.distanceM };
    facilityBaselineMean = isMine ? 22 : 38.4;
    facilityBaselineStd = isMine ? 6 : 8.0;
  } else {
    // Fall back to known legacy facilities catalog
    for (const f of KNOWN_FACILITIES) {
      const dist = haversine_m(detection.latitude, detection.longitude, f.lat, f.lon);
      if (!nearestFacility || dist < nearestFacility.distanceM) {
        nearestFacility = { name: f.name, distanceM: dist };
        if (dist <= f.radius_m) {
          facilityBaselineMean = f.baseline_frp_mean;
          facilityBaselineStd = f.baseline_frp_std;
        }
      }
    }
  }

  const isIndustrial = Boolean(industrialPolygon || (nearestFacility && nearestFacility.distanceM <= 3500));

  // 2. Protected area membership
  let inProtectedArea: { name: string; category: string } | null = null;
  for (const pa of PROTECTED_AREAS) {
    if (haversine_m(detection.latitude, detection.longitude, pa.lat, pa.lon) <= pa.radiusKm * 1000) {
      inProtectedArea = { name: pa.name, category: pa.category };
      break;
    }
  }

  const facilityZ = isIndustrial
    ? Number(((detection.frp - facilityBaselineMean) / Math.max(facilityBaselineStd, 0.1)).toFixed(2))
    : 0;

  const biome: HotspotTriage['context']['biome'] = isIndustrial
    ? 'industrial'
    : inProtectedArea
      ? 'protected-forest'
      : 'open-terrain';

  // 3. Class distribution: the real backend model when available, else a local heuristic
  // bucket table as an offline fallback (never trained, matches no ground truth — the
  // month-of-year "stubble season" branch below has no land/water awareness on its own).
  const month = detection.acquiredAt.getUTCMonth() + 1;
  let probs: Record<number, number>;
  let classId: number;
  let anomalyScore: number;
  let routeState: RouteState;
  let risk: RiskBreakdown;

  if (backend) {
    probs = backend.classProbabilities;
    classId = backend.classId;
    anomalyScore = backend.anomalyScore;
    routeState = backend.routeState;
    risk = backend.risk;
  } else {
    if (industrialPolygon?.isMine) {
      // Uncontrolled Mining / Coal-Seam Fire (Class 3)
      probs = { 1: 0.08, 2: 0.04, 3: 0.78, 4: 0.02, 5: 0.08 };
    } else if (isIndustrial && (facilityZ >= 4 || detection.frp >= 90)) {
      probs = { 1: 0.68, 2: 0.06, 3: 0.04, 4: 0.02, 5: 0.2 };
    } else if (isIndustrial) {
      probs = { 1: 0.08, 2: 0.03, 3: 0.03, 4: 0.02, 5: 0.84 };
    } else if (inProtectedArea && detection.frp >= 15) {
      probs = { 1: 0.03, 2: 0.82, 3: 0.03, 4: 0.09, 5: 0.03 };
    } else if (detection.frp >= 40 && detection.brightness >= 340) {
      probs = { 1: 0.05, 2: 0.7, 3: 0.05, 4: 0.17, 5: 0.03 };
    } else if (STUBBLE_MONTHS.has(month) && detection.frp < 30) {
      probs = { 1: 0.02, 2: 0.1, 3: 0.03, 4: 0.8, 5: 0.05 };
    } else {
      probs = { 1: 0.04, 2: 0.34, 3: 0.05, 4: 0.5, 5: 0.07 };
    }
    const total = Object.values(probs).reduce((s, v) => s + v, 0);
    Object.keys(probs).forEach((k) => (probs[Number(k)] = Number((probs[Number(k)] / total).toFixed(3))));

    classId = Number(Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0]);

    anomalyScore = Math.max(
      0.05,
      Math.min(0.98, (detection.frp / 120) * 0.5 + (facilityZ > 0 ? facilityZ * 0.08 : 0) + (detection.confidenceLevel === 'high' ? 0.15 : 0)),
    );

    routeState = routeEvent(
      probs,
      anomalyScore,
      isIndustrial,
      facilityZ,
      'multi_sensor_cross_confirmed',
      detection.confidenceLevel === 'high' ? 0.9 : detection.confidenceLevel === 'nominal' ? 0.7 : 0.45,
    );

    risk = computeRiskScore(
      probs,
      facilityZ,
      1,
      0,
      isIndustrial ? 0.7 : inProtectedArea ? 0.55 : 0.35,
    );
  }

  const windSpeed = weather?.windSpeedMps ?? 6.5;
  const windDir = weather?.windDirectionDeg ?? 135;
  const plume = generatePlumeCorridor([detection.latitude, detection.longitude], windSpeed, windDir);

  const explanation: string[] = [
    industrialPolygon
      ? industrialPolygon.isMine
        ? `Detection is inside mapped coal mining boundary: ${industrialPolygon.zone} (${industrialPolygon.name}) — coal-seam fire priority.`
        : `Detection is inside mapped industrial boundary: ${industrialPolygon.zone} (${industrialPolygon.name})${industrialPolygon.areaSqkm ? ` [${industrialPolygon.areaSqkm.toFixed(1)} km²]` : ''}${facilityZ ? ` (FRP ${facilityZ >= 0 ? '+' : ''}${facilityZ}σ vs baseline)` : ''}.`
      : isIndustrial
        ? `Detection lies ${(nearestFacility!.distanceM / 1000).toFixed(1)} km from ${nearestFacility!.name}${facilityZ ? ` (FRP ${facilityZ >= 0 ? '+' : ''}${facilityZ}σ vs facility baseline)` : ''}.`
        : nearestFacility
          ? `Nearest industrial facility (${nearestFacility.name}) is ${(nearestFacility.distanceM / 1000).toFixed(0)} km away — treated as open terrain.`
          : 'No industrial facility nearby — treated as open terrain.',
    inProtectedArea
      ? `Inside ${inProtectedArea.name} (${inProtectedArea.category}) — vegetation-fire priority.`
      : 'Not within a mapped protected area.',
    weather
      ? `Live atmospheric wind: ${weather.windSpeedMps.toFixed(1)} m/s toward ${weather.windDirectionDeg}° ${weather.cardinal} (gusts ${weather.windGustsMps.toFixed(1)} m/s, ${weather.temperatureC.toFixed(0)}°C) via Open-Meteo.`
      : 'Meteorological wind: default baseline (6.5 m/s @ 135° SE).',
    `Observed FRP ${detection.frp.toFixed(1)} MW, brightness ${detection.brightness.toFixed(0)} K, ${detection.daynight === 'D' ? 'daytime' : 'night-time'} overpass, ${detection.confidenceLevel} confidence.`,
    backend
      ? `Backend anomaly score ${anomalyScore.toFixed(2)} (${backend.anomalyModelVersion}); deterministic arbitration → ${routeState}.`
      : `Isolation-surrogate anomaly score ${anomalyScore.toFixed(2)}; deterministic arbitration → ${routeState}.`,
  ];

  if (backend && !backend.lulcInVocabulary) {
    explanation.push(
      'Land cover is outside the trained vocabulary for this point — classification relies mainly on ' +
        'radiometrics and industrial proximity, not land-cover context.',
    );
  }

  return {
    detection,
    context: { nearestFacility, industrialPolygon, inProtectedArea, isIndustrial, facilityZ, biome, weather },
    classProbabilities: probs,
    classId,
    className: CLASS_NAMES[classId],
    anomalyScore: Number(anomalyScore.toFixed(2)),
    routeState,
    risk,
    plume,
    explanation,
    disclaimer: backend
      ? `Classified by the trained CatBoost model (${backend.modelVersion}) via the Jvalyx backend, using live ` +
        'ESA WorldCover land cover and mapped industrial polygons.' +
        (backend.lulcInVocabulary
          ? ''
          : ' Land cover at this point fell outside the trained vocabulary, so land-cover-dependent signal is limited here.')
      : 'OFFLINE FALLBACK: Jvalyx triage is a heuristic lens over a single NASA FIRMS detection — not a trained ' +
        'model output. It classifies context (industrial vs forest vs cropland), not verified incident type.',
    source: backend ? 'model' : 'heuristic',
    lulcInVocabulary: backend?.lulcInVocabulary ?? true,
    modelVersion: backend?.modelVersion,
  };
}
