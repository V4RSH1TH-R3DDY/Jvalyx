/**
 * Client for the Jvalyx backend (FastAPI).
 *
 * `hooks/useBackendReplay.ts` drives the digital-twin dashboard from these calls plus
 * the `/ws/events` stream; `services/adapters.ts` reshapes the payloads into the
 * frontend `ScenarioFrame` vocabulary. When the backend is unreachable the dashboard
 * falls back to the bundled scenario pack and client-side math.
 */

const BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ??
  'http://localhost:8000';

export type BackendRouteState = 'NORMAL' | 'UNCERTAIN' | 'CRITICAL';
export type BackendMode = 'LIVE DATA' | 'HISTORICAL REPLAY' | 'DEMO SIMULATION MODE';

export interface BackendEvidenceCard {
  category: 'thermal' | 'context' | 'temporal' | 'decision';
  title: string;
  metrics: Record<string, string>;
  why_it_matters: string;
}

export interface BackendRiskBreakdown {
  total: number;
  severity: number;
  anomaly: number;
  spread: number;
  exposure: number;
}

export interface BackendEventIntelligence {
  event_id: string;
  scenario_id: string;
  frame_index: number;
  checkpoint: string | null;
  label: string;
  description: string;
  timestamp: string;
  mode: BackendMode;
  route_state: BackendRouteState;
  fused_event: Record<string, unknown>;
  features: Record<string, number>;
  decision: {
    class_id: number;
    class_name: string;
    class_probabilities: Record<string, number>;
    anomaly_score: number;
    route_state: BackendRouteState;
    risk_score: number;
    confidence_state: 'low' | 'medium' | 'high';
    explanation: string[];
    recommended_action: string;
    model_version: string;
    policy_version: string;
  };
  risk: BackendRiskBreakdown;
  evidence: BackendEvidenceCard[];
  tactical: Record<string, unknown> | null;
  historical_baseline_timeline: Array<Record<string, number | string>>;
  verification_status: 'unverified' | 'human_confirmed' | 'human_rejected';
  simulated: boolean;
  deviation: number;
}

export interface BackendScenarioSummary {
  scenario_id: string;
  title: string;
  description: string;
  mode: BackendMode;
  frame_count: number;
  checkpoints: string[];
  facility: Record<string, unknown> | null;
}

export interface BackendReplayStatus {
  scenario_id: string | null;
  scenario_title: string | null;
  mode: BackendMode;
  replay_status: 'IDLE' | 'PLAYING' | 'PAUSED' | 'COMPLETED';
  speed: number;
  frame_index: number;
  frame_count: number;
  clients: number;
  model_version: string;
  policy_version: string;
}

export interface BackendAuditEntry {
  id: string;
  event_id: string;
  timestamp: string;
  action: string;
  operator: string;
  notes: string;
  prior_route_state: BackendRouteState | null;
  new_route_state: BackendRouteState | null;
}

async function request<T>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const { timeoutMs, ...rest } = init ?? {};
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    signal: timeoutMs ? AbortSignal.timeout(timeoutMs) : undefined,
    ...rest,
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => response.statusText);
    throw new Error(`Jvalyx API ${response.status} on ${path}: ${detail}`);
  }
  return (await response.json()) as T;
}

export const jvalyxApi = {
  baseUrl: BASE_URL,

  health: () =>
    request<{ status: string; model_version: string; policy_version: string }>('/health', {
      timeoutMs: 2500,
    }),
  config: () => request<Record<string, unknown>>('/config'),

  listScenarios: () => request<BackendScenarioSummary[]>('/scenarios'),
  startScenario: (id: string) => request<BackendReplayStatus>(`/scenarios/${id}/start`, { method: 'POST' }),
  resetScenario: (id: string) => request<BackendReplayStatus>(`/scenarios/${id}/reset`, { method: 'POST' }),

  pause: () => request<BackendReplayStatus>('/replay/pause', { method: 'POST' }),
  resume: () => request<BackendReplayStatus>('/replay/resume', { method: 'POST' }),
  setSpeed: (speed: number) =>
    request<BackendReplayStatus>('/replay/speed', { method: 'POST', body: JSON.stringify({ speed }) }),
  jump: (checkpoint: string) =>
    request<BackendReplayStatus>('/replay/jump', { method: 'POST', body: JSON.stringify({ checkpoint }) }),
  step: (delta: 1 | -1) =>
    request<BackendReplayStatus>('/replay/step', { method: 'POST', body: JSON.stringify({ delta }) }),
  replayStatus: () => request<BackendReplayStatus>('/replay/status'),

  listEvents: () => request<BackendEventIntelligence[]>('/events'),
  getEvent: (id: string) => request<BackendEventIntelligence>(`/events/${id}`),
  simulate: (id: string, deviation: number) =>
    request<BackendEventIntelligence>(`/events/${id}/simulate`, {
      method: 'POST',
      body: JSON.stringify({ deviation }),
    }),
  verify: (id: string, decision: 'confirm' | 'reject', operator?: string, notes?: string) =>
    request<BackendEventIntelligence>(`/events/${id}/verify`, {
      method: 'POST',
      body: JSON.stringify({ decision, operator, notes }),
    }),
  audit: () => request<BackendAuditEntry[]>('/audit'),
  offshoreStatus: () =>
    request<{
      configured: boolean;
      type: 'postgresql' | 'http_rest' | 'none';
      target_url: string | null;
      local_total: number;
      synced_offshore: number;
      pending_offshore: number;
    }>('/audit/offshore-status'),
  syncOffshore: () =>
    request<{
      status: string;
      synced_now: number;
      configured: boolean;
      type: string;
      target_url: string | null;
      local_total: number;
      synced_offshore: number;
      pending_offshore: number;
    }>('/audit/offshore-sync', { method: 'POST' }),

  // -- Cell broadcast (see docs/cell-broadcast.md) --------------------------
  cbsStatus: () => request<BackendCbsStatus>('/cbs/status'),
  cbsBroadcast: (body: CbsBroadcastRequest) =>
    request<BackendCbsReceipt>('/cbs/broadcast', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
};

export interface BackendCbsStatus {
  devices_attached: number;
  located: number;
  cells: Record<string, number>;
  armed: boolean;
  venue: { latitude: number; longitude: number };
  ntfy_topic: string | null;
  last_alert: Record<string, unknown> | null;
}

export interface CbsBroadcastRequest {
  event_id?: string;
  route_state?: BackendRouteState;
  latitude?: number;
  longitude?: number;
  area_desc?: string;
  /** CAP `<event>` — carries the fire class, e.g. "Class 3 — Industrial Flare". */
  event?: string;
  headline?: string;
  radius_km?: number;
  status?: 'Actual' | 'Exercise' | 'Test';
}

export interface BackendCbsReceipt {
  identifier: string;
  severity: string;
  cmas_class: string;
  message_identifier: number;
  status: string;
  cells_in_footprint: string[];
  devices_attached: number;
  devices_in_footprint: number;
  ntfy_dispatched: boolean;
  cap_xml_url: string;
}

export type JvalyxSocketMessage =
  | { type: 'snapshot'; status: BackendReplayStatus; events: BackendEventIntelligence[] }
  | {
      type: 'event_update';
      timestamp: string;
      event_id: string;
      frame_index: number;
      checkpoint: string | null;
      changed: string[];
      payload: Record<string, unknown>;
    }
  | { type: 'replay_status'; reason: string; status: BackendReplayStatus };

const RECONNECT_BASE_DELAY_MS = 1000;
const RECONNECT_MAX_DELAY_MS = 15000;

/**
 * Opens the live event stream and keeps it open, reconnecting with exponential backoff
 * (capped at 15s) whenever the connection drops. `onStatusChange` fires on every
 * open/close transition so callers can reflect a dropped socket in the UI instead of
 * silently going stale. Returns a function that closes the stream and stops reconnecting.
 */
export function connectEventStream(
  onMessage: (message: JvalyxSocketMessage) => void,
  onStatusChange?: (open: boolean) => void,
): () => void {
  const wsUrl = `${BASE_URL.replace(/^http/, 'ws')}/ws/events`;
  let stopped = false;
  let socket: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let attempt = 0;

  const connect = () => {
    if (stopped) return;
    socket = new WebSocket(wsUrl);

    socket.addEventListener('open', () => {
      attempt = 0;
      onStatusChange?.(true);
    });
    socket.addEventListener('close', () => {
      onStatusChange?.(false);
      if (stopped) return;
      const delay = Math.min(RECONNECT_BASE_DELAY_MS * 2 ** attempt, RECONNECT_MAX_DELAY_MS);
      attempt += 1;
      reconnectTimer = setTimeout(connect, delay);
    });
    socket.addEventListener('message', (event) => {
      try {
        onMessage(JSON.parse(event.data) as JvalyxSocketMessage);
      } catch {
        /* ignore malformed frames */
      }
    });
  };

  connect();

  return () => {
    stopped = true;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    socket?.close();
  };
}
