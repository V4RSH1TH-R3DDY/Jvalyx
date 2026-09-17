import React from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  Flame,
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  ClipboardList,
  SkipBack,
  SkipForward,
  Radio,
  PlugZap,
} from 'lucide-react';
import type { RouteState } from '../types';
import { CellBroadcastButton } from './CellBroadcastButton';

export interface HeaderConnection {
  status: 'connecting' | 'online' | 'offline';
  socketOpen: boolean;
  modelVersion: string;
  policyVersion: string;
  error: string | null;
}

interface HeaderProps {
  scenarios: { id: string; name: string; category: string; frameCount?: number }[];
  /** Demo honesty label (Comprehensive plan §18) — always visible, never inferred. */
  dataMode: 'LIVE DATA' | 'HISTORICAL REPLAY' | 'DEMO SIMULATION MODE';
  currentScenarioId: string;
  currentFrameIndex: number;
  frameCount: number;
  isPlaying: boolean;
  playbackSpeed: number;
  routeState: RouteState;
  connection: HeaderConnection;
  checkpoints: string[];
  onSelectScenario: (scenarioId: string) => void;
  onTogglePlay: () => void;
  onReset: () => void;
  onSpeedChange: (speed: number) => void;
  onStep: (delta: 1 | -1) => void;
  /** Checkpoint jumps are backend-only; omitted in offline mode. */
  onJump?: (checkpoint: string) => void;
  onOpenAuditLog: () => void;
}

const ROUTE_BADGES: Record<RouteState, { bg: string; icon: typeof ShieldAlert; dot: string; label: string }> = {
  CRITICAL: {
    bg: 'bg-rose-950/40 border-rose-800/80 text-rose-300',
    icon: ShieldAlert,
    dot: 'bg-rose-500 animate-ping',
    label: 'CRITICAL ESCALATION',
  },
  UNCERTAIN: {
    bg: 'bg-amber-950/40 border-amber-800/80 text-amber-300',
    icon: AlertTriangle,
    dot: 'bg-amber-500 animate-pulse',
    label: 'UNCERTAIN — VERIFY',
  },
  NORMAL: {
    bg: 'bg-zinc-900 border-zinc-700 text-zinc-300',
    icon: CheckCircle2,
    dot: 'bg-zinc-400',
    label: 'NORMAL ROUTINE',
  },
};

/** Where the frame on screen came from — never leave this ambiguous during a demo. */
const ConnectionBadge: React.FC<{ connection: HeaderConnection }> = ({ connection }) => {
  const { status, socketOpen, modelVersion, policyVersion, error } = connection;

  const style =
    status === 'online'
      ? 'border-emerald-900/60 bg-emerald-950/30 text-emerald-400'
      : status === 'connecting'
        ? 'border-zinc-700 bg-zinc-900 text-zinc-400'
        : 'border-amber-900/60 bg-amber-950/30 text-amber-400';

  const label =
    status === 'online'
      ? socketOpen
        ? 'LIVE BACKEND · WS'
        : 'LIVE BACKEND · POLLING'
      : status === 'connecting'
        ? 'CONNECTING…'
        : 'OFFLINE PACK';

  const Icon = status === 'online' ? Radio : PlugZap;

  return (
    <div
      title={
        status === 'offline'
          ? `Backend unreachable${error ? `: ${error}` : ''} — replaying the bundled pack with client-side math.`
          : `model ${modelVersion} · policy ${policyVersion}`
      }
      className={`flex items-center gap-1.5 border px-2.5 py-1.5 font-mono text-[10px] font-bold tracking-wider ${style}`}
    >
      <Icon className="h-3.5 w-3.5" />
      <span>{label}</span>
    </div>
  );
};

export const Header: React.FC<HeaderProps> = ({
  scenarios,
  currentScenarioId,
  currentFrameIndex,
  frameCount,
  dataMode,
  isPlaying,
  playbackSpeed,
  routeState,
  connection,
  checkpoints,
  onSelectScenario,
  onTogglePlay,
  onReset,
  onSpeedChange,
  onStep,
  onJump,
  onOpenAuditLog,
}) => {
  const badge = ROUTE_BADGES[routeState];
  const BadgeIcon = badge.icon;

  return (
    <header className="w-full select-none border-b border-border bg-background-subtle text-zinc-100">
      <div className="flex flex-wrap items-center justify-between gap-4 px-5 py-2.5">
        {/* View title & scenario picker */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 border border-zinc-700 bg-zinc-900 px-3 py-1.5 shadow-solid-sm">
            <Flame className="h-5 w-5 fill-rose-500/20 text-rose-500" />
            <span className="font-mono text-xs font-black tracking-widest text-zinc-100">
              INCIDENT DIGITAL TWIN
            </span>
          </div>

          <div className="flex items-center gap-2">
            <label className="font-mono text-xs font-semibold uppercase text-zinc-400">
              SCENARIO:
            </label>
            <select
              value={currentScenarioId}
              onChange={(e) => onSelectScenario(e.target.value)}
              className="cursor-pointer border border-border bg-background-card px-3 py-1.5 font-sans text-xs font-medium text-zinc-100 hover:border-border-highlight focus:border-cyan-500 focus:outline-none"
            >
              {scenarios.map((sc) => (
                <option key={sc.id} value={sc.id} className="bg-background text-zinc-100">
                  [{sc.category.toUpperCase()}] {sc.name}{sc.frameCount ? ` (${sc.frameCount} step${sc.frameCount > 1 ? 's' : ''})` : ''}
                </option>
              ))}
            </select>
          </div>

          <span
            title="Data mode reported by the pipeline"
            className={`border px-2 py-0.5 font-mono text-[10px] font-bold tracking-wider ${
              dataMode === 'DEMO SIMULATION MODE'
                ? 'animate-pulse border-amber-500/80 bg-amber-950/60 text-amber-300'
                : dataMode === 'LIVE DATA'
                  ? 'border-emerald-500/80 bg-emerald-950/60 text-emerald-300'
                  : 'border-border bg-background-card text-zinc-400'
            }`}
          >
            {dataMode}
          </span>
        </div>

        {/* Replay transport */}
        <div className="flex items-center gap-2 border border-border bg-background px-2.5 py-1">
          <button
            onClick={() => onStep(-1)}
            disabled={currentFrameIndex <= 0}
            title="Step back one frame"
            className="border border-border bg-background-card p-1.5 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <SkipBack className="h-3.5 w-3.5" />
          </button>

          <button
            onClick={onTogglePlay}
            disabled={frameCount <= 1}
            title={frameCount <= 1 ? 'Single-frame snapshot' : isPlaying ? 'Pause' : 'Play timeline'}
            className="flex items-center gap-1.5 border border-border bg-background-card px-3 py-1 font-mono text-xs font-bold text-zinc-200 transition-colors hover:border-cyan-500 hover:bg-cyan-950 hover:text-cyan-300 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isPlaying ? (
              <>
                <Pause className="h-3.5 w-3.5 text-amber-400" />
                <span>PAUSE</span>
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5 fill-cyan-400 text-cyan-400" />
                <span>{frameCount <= 1 ? 'STATIC FRAME' : 'PLAY REPLAY'}</span>
              </>
            )}
          </button>

          <button
            onClick={() => onStep(1)}
            disabled={currentFrameIndex >= frameCount - 1}
            title="Step forward one frame"
            className="border border-border bg-background-card p-1.5 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <SkipForward className="h-3.5 w-3.5" />
          </button>

          <button
            onClick={onReset}
            title="Reset timeline"
            className="border border-border bg-background-card p-1.5 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </button>

          <div className="flex items-center border border-border bg-background-card font-mono text-[11px]">
            {[0.5, 1, 4].map((speed) => (
              <button
                key={speed}
                onClick={() => onSpeedChange(speed)}
                className={`border-r border-border px-2 py-0.5 transition-colors last:border-r-0 ${
                  playbackSpeed === speed
                    ? 'bg-zinc-700 font-bold text-zinc-100'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                {speed}x
              </button>
            ))}
          </div>

          {onJump && checkpoints.length > 0 && (
            <select
              value=""
              onChange={(e) => {
                if (e.target.value) onJump(e.target.value);
                e.target.value = '';
              }}
              title="Jump to a scenario checkpoint"
              className="cursor-pointer border border-border bg-background-card px-2 py-1 font-mono text-[11px] text-zinc-300 hover:border-border-highlight focus:border-zinc-500 focus:outline-none"
            >
              <option value="">JUMP TO…</option>
              {checkpoints.map((cp) => (
                <option key={cp} value={cp} className="bg-background">
                  {cp.replace(/_/g, ' ')}
                </option>
              ))}
            </select>
          )}

          <div className="ml-1 flex items-center gap-1.5 border-l border-border px-2 font-mono text-xs text-zinc-400">
            <span>STEP:</span>
            <span className="font-tabular font-semibold text-zinc-100">
              {Math.min(currentFrameIndex + 1, frameCount)}/{frameCount}
            </span>
          </div>
        </div>

        {/* Provenance, route badge & audit log */}
        <div className="flex items-center gap-3">
          <ConnectionBadge connection={connection} />

          <div className={`flex items-center gap-2 border px-3 py-1.5 transition-all ${badge.bg}`}>
            <BadgeIcon className="h-4 w-4 shrink-0" />
            <span className="font-mono text-xs font-bold tracking-wider">{badge.label}</span>
          </div>

          <CellBroadcastButton />

          <button
            onClick={onOpenAuditLog}
            className="flex items-center gap-1.5 border border-border bg-background-card px-2.5 py-1.5 font-mono text-xs text-zinc-300 transition-colors hover:border-zinc-500 hover:bg-zinc-800 hover:text-zinc-100"
          >
            <ClipboardList className="h-3.5 w-3.5" />
            AUDIT LOG
          </button>
        </div>
      </div>
    </header>
  );
};
