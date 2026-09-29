/**
 * Offline fallback: the bundled scenario pack replayed with client-side math.
 *
 * Used only when the backend is unreachable. While it runs, arbitration, risk and the
 * counterfactual are recomputed in the browser (`utils/math.ts`) and the audit log lives
 * in React state, so the dashboard labels the source as OFFLINE PACK.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { SCENARIOS } from '../data/scenarios';
import type { OperatorAuditEntry, RouteState, Scenario, ScenarioFrame } from '../types';
import { computeRiskScore, DEFAULT_ARBITRATION_CONFIG, generatePlumeCorridor, routeEvent } from '../utils/math';

const SEED_AUDIT: OperatorAuditEntry[] = [
  {
    id: 'aud-init-001',
    eventId: 'evt-init-000',
    timestamp: '2026-09-05 09:30:00 UTC',
    action: 'REQUEST_TACTICAL_PASS',
    operator: 'OPS-DUTY-1',
    notes: 'Initial mission baseline validated against historical catalog.',
    priorRouteState: 'NORMAL',
    newRouteState: 'NORMAL',
  },
];

export function useLocalReplay(enabled: boolean) {
  const [selectedScenarioId, setSelectedScenarioId] = useState('industrial_escalation');
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [deviation, setDeviation] = useState(0);
  const [isSimulated, setIsSimulated] = useState(false);
  const [manualRouteOverride, setManualRouteOverride] = useState<RouteState | null>(null);
  const [auditEntries, setAuditEntries] = useState<OperatorAuditEntry[]>(SEED_AUDIT);

  const scenario: Scenario = useMemo(
    () => SCENARIOS.find((s) => s.id === selectedScenarioId) || SCENARIOS[0],
    [selectedScenarioId],
  );

  const resetSimulation = useCallback(() => {
    setDeviation(0);
    setIsSimulated(false);
    setManualRouteOverride(null);
  }, []);

  const selectScenario = useCallback(
    (scenarioId: string) => {
      setSelectedScenarioId(scenarioId);
      setCurrentFrameIndex(0);
      setIsPlaying(false);
      resetSimulation();
    },
    [resetSimulation],
  );

  const reset = useCallback(() => {
    setCurrentFrameIndex(0);
    setIsPlaying(false);
    resetSimulation();
  }, [resetSimulation]);

  const step = useCallback(
    (delta: 1 | -1) => {
      setIsPlaying(false);
      setCurrentFrameIndex((prev) =>
        Math.max(0, Math.min(prev + delta, scenario.frames.length - 1)),
      );
    },
    [scenario.frames.length],
  );

  // Replay clock
  useEffect(() => {
    if (!enabled || !isPlaying) return;
    const delay = Math.max(800, 3000 / playbackSpeed);
    const interval = setInterval(() => {
      setCurrentFrameIndex((prev) => {
        if (prev >= scenario.frames.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, delay);
    return () => clearInterval(interval);
  }, [enabled, isPlaying, playbackSpeed, scenario.frames.length]);

  const baseFrame: ScenarioFrame = useMemo(
    () => scenario.frames[currentFrameIndex] || scenario.frames[0],
    [scenario, currentFrameIndex],
  );

  const frame: ScenarioFrame = useMemo(() => {
    if (!isSimulated && deviation === 0) {
      if (manualRouteOverride) {
        return {
          ...baseFrame,
          decision: { ...baseFrame.decision, route_state: manualRouteOverride },
          fusedEvent: { ...baseFrame.fusedEvent, route_state: manualRouteOverride },
        };
      }
      return baseFrame;
    }

    const baseFRP = baseFrame.fusedEvent.detections.reduce((a, b) => a + b.frp_mw, 0);
    const mean = baseFrame.fusedEvent.baseline_frp_mean;
    const std = baseFrame.fusedEvent.baseline_frp_std;

    const simFRP = baseFRP + deviation * 250;
    const simZ = Number(((simFRP - mean) / Math.max(std, 0.1)).toFixed(2));
    const simCluster = Math.min(8, Math.max(1, Math.round(1 + deviation * 6)));

    const p1 = Math.min(0.96, Math.max(0.05, 0.1 + deviation * 0.85));
    const p5 = Math.max(0.01, 1 - p1 - 0.05);
    const simProbs = {
      1: Number(p1.toFixed(2)),
      2: Number((0.03 + deviation * 0.02).toFixed(2)),
      3: 0.01,
      4: 0.01,
      5: Number(p5.toFixed(2)),
    };
    const simAnomaly = Math.min(0.98, Math.max(0.1, 0.15 + deviation * 0.82));

    const activeRoute =
      manualRouteOverride ||
      routeEvent(
        simProbs,
        simAnomaly,
        baseFrame.fusedEvent.is_in_industrial_polygon,
        simZ,
        baseFrame.fusedEvent.sensor_agreement_state,
        baseFrame.fusedEvent.data_quality_score,
      );

    const simRisk = computeRiskScore(
      simProbs,
      simZ,
      simCluster,
      baseFrame.fusedEvent.centroid_drift_velocity_mph,
      0.75,
    );

    const basePlume = baseFrame.tactical?.plumeCorridor;
    const simPlume = generatePlumeCorridor(
      [baseFrame.fusedEvent.latitude, baseFrame.fusedEvent.longitude],
      basePlume?.windSpeedMps || 7.0,
      basePlume?.windDirectionDeg || 135,
    );

    return {
      ...baseFrame,
      fusedEvent: {
        ...baseFrame.fusedEvent,
        facility_frp_zscore: simZ,
        frp_z_score: simZ,
        cluster_pixel_count: simCluster,
        route_state: activeRoute,
        detections: [{ ...baseFrame.fusedEvent.detections[0], frp_mw: simFRP }],
      },
      decision: {
        ...baseFrame.decision,
        class_id: simProbs[1] >= DEFAULT_ARBITRATION_CONFIG.class1_threshold ? 1 : baseFrame.decision.class_id,
        class_name:
          simProbs[1] >= DEFAULT_ARBITRATION_CONFIG.class1_threshold
            ? 'Unusual Industrial Fire'
            : baseFrame.decision.class_name,
        class_probabilities: simProbs,
        anomaly_score: simAnomaly,
        route_state: activeRoute,
        risk_score: simRisk.total,
        explanation: [
          `What-If Simulation: Operational deviation set to ${deviation.toFixed(2)}.`,
          `FRP shifted to ${simFRP.toFixed(1)} MW (Z = ${simZ}σ relative to normal baseline).`,
        ],
      },
      risk: simRisk,
      tactical: {
        ...baseFrame.tactical,
        plumeCorridor: simPlume,
        affectedAssets: baseFrame.tactical?.affectedAssets || scenario.facility.nearbyAssets,
      },
      historicalBaselineTimeline: baseFrame.historicalBaselineTimeline.map((pt, i) =>
        i === baseFrame.historicalBaselineTimeline.length - 1 ? { ...pt, observedFRP: simFRP } : pt,
      ),
    };
  }, [baseFrame, deviation, isSimulated, manualRouteOverride, scenario.facility.nearbyAssets]);

  const simulate = useCallback((value: number) => {
    setDeviation(value);
    setIsSimulated(true);
  }, []);

  const verify = useCallback(
    (action: 'CONFIRM_CRITICAL' | 'REJECT_NORMAL') => {
      const prior = frame.decision.route_state;
      const next: RouteState = action === 'CONFIRM_CRITICAL' ? 'CRITICAL' : 'NORMAL';
      setManualRouteOverride(next);
      setAuditEntries((prev) => [
        {
          id: `aud-${Date.now()}`,
          eventId: frame.fusedEvent.event_id,
          timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC',
          action,
          operator: 'DUTY-SUPERVISOR-1',
          notes:
            action === 'CONFIRM_CRITICAL'
              ? 'Operator confirmed thermal anomaly escalation. Tactical analysis tasking issued.'
              : 'Operator verified routine operational heat signature. Escalation suppressed.',
          priorRouteState: prior,
          newRouteState: next,
        },
        ...prev,
      ]);
    },
    [frame],
  );

  return {
    scenario,
    scenarios: SCENARIOS,
    frame,
    frameIndex: currentFrameIndex,
    frameCount: scenario.frames.length,
    isPlaying,
    playbackSpeed,
    deviation,
    isSimulated,
    auditEntries,
    checkpoints: [] as string[],
    actions: {
      selectScenario,
      togglePlay: () => {
        setIsPlaying((p) => {
          if (!p && currentFrameIndex >= scenario.frames.length - 1) {
            setCurrentFrameIndex(0);
          }
          return !p;
        });
      },
      reset,
      setSpeed: setPlaybackSpeed,
      step,
      simulate,
      resetSimulation,
      verify,
    },
  };
}
