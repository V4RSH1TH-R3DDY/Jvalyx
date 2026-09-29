/**
 * Drives the digital-twin dashboard from the FastAPI backend.
 *
 * The Python runtime owns the replay clock, arbitration, risk and the audit log; this
 * hook subscribes to `/ws/events` and mirrors that state into React. Playback controls,
 * the counterfactual slider and operator verification are all round trips to the API —
 * no client-side recompute while `status === 'online'`.
 *
 * If the backend cannot be reached the hook reports `offline` and the dashboard falls
 * back to the bundled scenario pack (`hooks/useLocalReplay.ts`).
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  connectEventStream,
  jvalyxApi,
  type BackendEventIntelligence,
  type BackendReplayStatus,
  type BackendScenarioSummary,
} from '../services/api';
import { toAuditEntries, toFacility, toScenarioFrame } from '../services/adapters';
import type { FacilityDigitalTwin, OperatorAuditEntry, ScenarioFrame } from '../types';

export type BackendStatus = 'connecting' | 'online' | 'offline';

export interface BackendReplay {
  status: BackendStatus;
  socketOpen: boolean;
  error: string | null;
  scenarios: BackendScenarioSummary[];
  scenarioId: string | null;
  replayStatus: BackendReplayStatus | null;
  event: BackendEventIntelligence | null;
  frame: ScenarioFrame | null;
  facility: FacilityDigitalTwin | undefined;
  auditEntries: OperatorAuditEntry[];
  deviation: number;
  isSimulated: boolean;
  isPlaying: boolean;
  checkpoints: string[];
  actions: {
    selectScenario: (scenarioId: string) => Promise<void>;
    togglePlay: () => Promise<void>;
    reset: () => Promise<void>;
    setSpeed: (speed: number) => Promise<void>;
    step: (delta: 1 | -1) => Promise<void>;
    jump: (checkpoint: string) => Promise<void>;
    simulate: (deviation: number) => Promise<void>;
    resetSimulation: () => Promise<void>;
    verify: (action: 'CONFIRM_CRITICAL' | 'REJECT_NORMAL') => Promise<void>;
    refreshAudit: () => Promise<void>;
  };
}

/** `enabled` lets the caller skip all backend traffic (e.g. user forced offline mode). */
export function useBackendReplay(
  enabled: boolean,
  facilityFallbacks: Record<string, FacilityDigitalTwin>,
): BackendReplay {
  const [probeStatus, setStatus] = useState<BackendStatus>('connecting');
  // A caller-disabled hook is offline by definition; no need to store that separately.
  const status: BackendStatus = enabled ? probeStatus : 'offline';
  const [socketOpen, setSocketOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scenarios, setScenarios] = useState<BackendScenarioSummary[]>([]);
  const [replayStatus, setReplayStatus] = useState<BackendReplayStatus | null>(null);
  const [event, setEvent] = useState<BackendEventIntelligence | null>(null);
  const [auditEntries, setAuditEntries] = useState<OperatorAuditEntry[]>([]);
  // Optimistic slider position: the server value round-trips too slowly to drive the
  // input directly while the operator is dragging it.
  const [pendingDeviation, setPendingDeviation] = useState<number | null>(null);
  const simulateTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Guards a race: a frame that arrives from the socket must not clobber a newer
  // simulated/verified event we just received from a REST round trip.
  const latestWrite = useRef(0);

  const applyEvent = useCallback((next: BackendEventIntelligence | null, stamp = Date.now()) => {
    if (stamp < latestWrite.current) return;
    latestWrite.current = stamp;
    setEvent(next);
  }, []);

  const refreshAudit = useCallback(async () => {
    try {
      setAuditEntries(toAuditEntries(await jvalyxApi.audit()));
    } catch {
      /* audit is non-critical; keep the last good log */
    }
  }, []);

  // -- connect: health -> catalog -> current state -----------------------
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;

    (async () => {
      try {
        await jvalyxApi.health();
        const [catalog, replay, events] = await Promise.all([
          jvalyxApi.listScenarios(),
          jvalyxApi.replayStatus(),
          jvalyxApi.listEvents(),
        ]);
        if (cancelled) return;
        setScenarios(catalog);
        setReplayStatus(replay);
        applyEvent(events[0] ?? null);
        setStatus('online');
        setError(null);
        await refreshAudit();

        // Nothing loaded yet: seed the first scenario so the panel is never empty.
        if (!replay.scenario_id && catalog.length > 0) {
          setReplayStatus(await jvalyxApi.resetScenario(catalog[0].scenario_id));
          const seeded = await jvalyxApi.listEvents();
          if (!cancelled) applyEvent(seeded[0] ?? null);
        }
      } catch (err) {
        if (cancelled) return;
        setStatus('offline');
        setError(err instanceof Error ? err.message : String(err));
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [enabled, applyEvent, refreshAudit]);

  // -- live stream --------------------------------------------------------
  useEffect(() => {
    if (!enabled || status !== 'online') return;

    const close = connectEventStream(
      (message) => {
        if (message.type === 'snapshot') {
          setReplayStatus(message.status);
          applyEvent(message.events[0] ?? null);
          return;
        }
        if (message.type === 'replay_status') {
          setReplayStatus(message.status);
          return;
        }
        if (message.type === 'event_update') {
          // The update frame is a delta; pull the full event the backend just wrote.
          const stamp = Date.now();
          jvalyxApi
            .getEvent(message.event_id)
            .then((full) => applyEvent(full, stamp))
            .catch(() => undefined);
          setReplayStatus((prev) => (prev ? { ...prev, frame_index: message.frame_index } : prev));
        }
      },
      setSocketOpen,
    );

    return close;
  }, [enabled, status, applyEvent]);

  // -- polling fallback: keeps the frame fresh while the socket is reconnecting --------
  useEffect(() => {
    if (!enabled || status !== 'online' || socketOpen) return;

    let cancelled = false;
    const poll = async () => {
      try {
        const [replay, events] = await Promise.all([
          jvalyxApi.replayStatus(),
          jvalyxApi.listEvents(),
        ]);
        if (cancelled) return;
        setReplayStatus(replay);
        applyEvent(events[0] ?? null);
      } catch {
        /* the socket reconnect loop will restore the live stream; keep polling */
      }
    };

    const interval = setInterval(poll, 4000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [enabled, status, socketOpen, applyEvent]);

  useEffect(
    () => () => {
      if (simulateTimer.current) clearTimeout(simulateTimer.current);
    },
    [],
  );

  // -- actions -------------------------------------------------------------
  const run = useCallback(
    async (task: () => Promise<BackendReplayStatus | void>, refetchEvent = true) => {
      try {
        const stamp = Date.now();
        const next = await task();
        if (next) setReplayStatus(next);
        if (refetchEvent) {
          const events = await jvalyxApi.listEvents();
          applyEvent(events[0] ?? null, stamp);
        }
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [applyEvent],
  );

  const scenarioId = replayStatus?.scenario_id ?? null;
  const isPlaying = replayStatus?.replay_status === 'PLAYING';

  const actions = useMemo(
    () => ({
      selectScenario: (id: string) => {
        setPendingDeviation(null);
        return run(() => jvalyxApi.resetScenario(id));
      },
      togglePlay: () =>
        run(async () => {
          const current = replayStatus?.replay_status;
          if (current === 'PLAYING') return jvalyxApi.pause();
          if (current === 'PAUSED') return jvalyxApi.resume();
          return scenarioId ? jvalyxApi.startScenario(scenarioId) : undefined;
        }, false),
      reset: () => {
        setPendingDeviation(null);
        return run(() => (scenarioId ? jvalyxApi.resetScenario(scenarioId) : Promise.resolve()));
      },
      setSpeed: (speed: number) => run(() => jvalyxApi.setSpeed(speed), false),
      step: (delta: 1 | -1) => {
        setPendingDeviation(null);
        return run(() => jvalyxApi.step(delta));
      },
      jump: (checkpoint: string) => run(() => jvalyxApi.jump(checkpoint)),
      simulate: async (deviation: number) => {
        setPendingDeviation(deviation);
        if (simulateTimer.current) clearTimeout(simulateTimer.current);
        simulateTimer.current = setTimeout(() => {
          void run(async () => {
            const id = event?.event_id;
            if (!id) return;
            applyEvent(await jvalyxApi.simulate(id, deviation));
          }, false);
        }, 140);
      },
      resetSimulation: () => {
        setPendingDeviation(0);
        if (simulateTimer.current) clearTimeout(simulateTimer.current);
        return run(async () => {
          const id = event?.event_id;
          if (!id) return;
          // deviation 0 recomputes the frame from the unmodified model output.
          applyEvent(await jvalyxApi.simulate(id, 0));
        }, false);
      },
      verify: async (action: 'CONFIRM_CRITICAL' | 'REJECT_NORMAL') => {
        const id = event?.event_id;
        if (!id) return;
        await run(async () => {
          applyEvent(
            await jvalyxApi.verify(id, action === 'CONFIRM_CRITICAL' ? 'confirm' : 'reject'),
          );
        }, false);
        await refreshAudit();
      },
      refreshAudit,
    }),
    [run, replayStatus, scenarioId, event, applyEvent, refreshAudit],
  );

  const facility = useMemo(() => {
    const summary = scenarios.find((s) => s.scenario_id === (event?.scenario_id ?? scenarioId));
    return toFacility(summary, facilityFallbacks[event?.scenario_id ?? scenarioId ?? '']);
  }, [scenarios, event, scenarioId, facilityFallbacks]);

  const frame = useMemo(() => (event ? toScenarioFrame(event, facility) : null), [event, facility]);

  const checkpoints = useMemo(
    () => scenarios.find((s) => s.scenario_id === scenarioId)?.checkpoints ?? [],
    [scenarios, scenarioId],
  );

  return {
    status,
    socketOpen,
    error,
    scenarios,
    scenarioId,
    replayStatus,
    event,
    frame,
    facility,
    auditEntries,
    deviation: pendingDeviation ?? event?.deviation ?? 0,
    isSimulated: (pendingDeviation ?? 0) > 0 || Boolean(event?.simulated),
    isPlaying,
    checkpoints,
    actions,
  };
}
