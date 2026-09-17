import { useEffect, useMemo, useState } from 'react';
import {
  X,
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  Flame,
  Wind,
  Compass,
  Droplets,
  Thermometer,
} from 'lucide-react';
import { useFires, useFiresDispatch, useVisibleDetections } from '../state/store';
import { triageHotspot } from '../analysis/triage';
import { classifyDetectionBackend, type BackendClassification } from '../analysis/backendClassify';
import { setClassification } from '../analysis/classificationCache';
import { fetchLiveWeather, type LiveWeatherData } from '../../services/weather';
import { FIRE_CLASSES } from '../../data/scenarios';
import type { FireClassId, RouteState } from '../../types';
import { cn } from './ui';
import { FireAlertBroadcast } from './FireAlertBroadcast';
import { FIRE_CLASS_PNG, CLASS_ID_TO_KEY } from '../analysis/iconMap';

const ROUTE_STYLE: Record<RouteState, { bg: string; icon: typeof ShieldAlert; label: string }> = {
  CRITICAL: { bg: 'bg-rose-950/60 border-rose-500 text-rose-200', icon: ShieldAlert, label: 'Critical' },
  UNCERTAIN: { bg: 'bg-amber-950/60 border-amber-500 text-amber-200', icon: AlertTriangle, label: 'Uncertain — verify' },
  NORMAL: { bg: 'bg-cyan-950/60 border-cyan-500 text-cyan-200', icon: CheckCircle2, label: 'Normal / routine' },
};

export function HotspotAnalysis() {
  const { analysisOpen, selectedId } = useFires();
  const dispatch = useFiresDispatch();
  const visible = useVisibleDetections();
  const detection = visible.find((d) => d.id === selectedId);

  const [weather, setWeather] = useState<LiveWeatherData | null>(null);
  const [loadingWeather, setLoadingWeather] = useState(false);

  useEffect(() => {
    if (!detection) {
      setWeather(null);
      return;
    }
    let active = true;
    setLoadingWeather(true);
    fetchLiveWeather(detection.latitude, detection.longitude)
      .then((w) => {
        if (active) {
          setWeather(w);
          setLoadingWeather(false);
        }
      })
      .catch(() => {
        if (active) setLoadingWeather(false);
      });
    return () => {
      active = false;
    };
  }, [detection?.id, detection?.latitude, detection?.longitude]);

  const [backendClassification, setBackendClassification] = useState<BackendClassification | null>(null);
  const [backendUnavailable, setBackendUnavailable] = useState(false);

  useEffect(() => {
    if (!detection) {
      setBackendClassification(null);
      setBackendUnavailable(false);
      return;
    }
    let active = true;
    const controller = new AbortController();
    setBackendClassification(null);
    classifyDetectionBackend(detection, controller.signal)
      .then((result) => {
        if (active) {
          setBackendClassification(result);
          setBackendUnavailable(false);
          // Persist to cache so FireMap can update the map marker icon immediately
          setClassification(detection.id, { classId: result.classId, routeState: result.routeState });
        }
      })
      .catch(() => {
        // Backend unreachable/erroring — triageHotspot() below falls back to its local
        // heuristic exactly like the digital twin falls back to the offline replay pack.
        if (active) setBackendUnavailable(true);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [detection?.id, detection?.latitude, detection?.longitude]);

  const triage = useMemo(
    () => (detection ? triageHotspot(detection, weather ?? undefined, backendClassification ?? undefined) : null),
    [detection, weather, backendClassification],
  );

  if (!analysisOpen) return null;

  return (
    <>
      <div
        className="fixed inset-0 z-[1500] bg-black/50"
        onClick={() => dispatch({ type: 'closeAnalysis' })}
      />
      <aside className="fixed right-0 top-0 z-[1501] flex h-full w-[420px] max-w-[92vw] flex-col bg-[#0b0f14] text-white shadow-2xl">
        <div className="flex items-center justify-between border-b border-white/10 bg-gradient-to-r from-[#7a0c28] to-[#a11540] px-4 py-3">
          <span className="flex items-center gap-2 text-sm font-black uppercase tracking-widest">
            <Flame className="h-4 w-4" /> Jvalyx Hotspot Triage
          </span>
          <button type="button" onClick={() => dispatch({ type: 'closeAnalysis' })}>
            <X className="h-5 w-5" />
          </button>
        </div>

        {!triage ? (
          <div className="p-6 text-sm text-white/50">Select a fire detection on the map to analyse it.</div>
        ) : (
          <div className="flex-1 space-y-3 overflow-y-auto p-4 text-xs">
            <div className="flex items-center justify-end">
              <span
                className={cn(
                  'rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider',
                  triage.source === 'model'
                    ? 'bg-emerald-950/60 text-emerald-300 border border-emerald-500/40'
                    : 'bg-zinc-800/60 text-zinc-400 border border-zinc-600/40',
                )}
              >
                {triage.source === 'model' ? 'Live model' : backendUnavailable ? 'Offline heuristic (backend unreachable)' : 'Offline heuristic'}
              </span>
            </div>

            <RouteBanner route={triage.routeState} />

            {detection && (
              <FireAlertBroadcast
                latitude={detection.latitude}
                longitude={detection.longitude}
                classId={triage.classId}
                className={triage.className}
                routeState={triage.routeState}
                placeName={triage.context.industrialPolygon?.name}
              />
            )}

            {triage.context.industrialPolygon && (
              <div className="rounded border border-amber-500/40 bg-amber-950/40 p-2.5">
                <div className="flex items-center gap-1.5 font-bold uppercase tracking-wider text-amber-300 text-[11px]">
                  <Flame className="h-3.5 w-3.5 text-amber-400 shrink-0" />
                  {triage.context.industrialPolygon.isMine ? 'Coalfield / Mining Basin' : 'Industrial Complex Boundary'}
                </div>
                <div className="mt-1 text-xs font-bold text-zinc-100">
                  {triage.context.industrialPolygon.name}
                </div>
                <div className="mt-1 flex items-center justify-between text-[10px] text-zinc-400 font-mono">
                  <span className="text-amber-200/80">{triage.context.industrialPolygon.zone}</span>
                  {triage.context.industrialPolygon.areaSqkm && (
                    <span>{triage.context.industrialPolygon.areaSqkm.toFixed(2)} km²</span>
                  )}
                </div>
              </div>
            )}

            <Section title={triage.source === 'model' ? 'Classification (trained CatBoost model)' : 'Classification (offline heuristic lens)'}>
              <div className="mb-2 flex items-center gap-2">
                <img
                  src={FIRE_CLASS_PNG[CLASS_ID_TO_KEY[triage.classId] ?? 'industrial']}
                  alt={triage.className}
                  width={20}
                  height={20}
                  className="shrink-0 object-contain"
                  onError={(e) => {
                    (e.currentTarget as HTMLElement).style.display = 'none';
                  }}
                />
                <span className="font-bold uppercase tracking-wide">{triage.className}</span>
              </div>
              <div className="space-y-1">
                {([1, 2, 3, 4, 5] as FireClassId[]).map((c) => {
                  const p = triage.classProbabilities[c] ?? 0;
                  return (
                    <div key={c} className="flex items-center gap-2">
                      <span className="w-6 text-white/50">C{c}</span>
                      <div className="h-2 flex-1 overflow-hidden bg-white/10">
                        <div
                          className="h-full"
                          style={{
                            width: `${Math.round(p * 100)}%`,
                            background: FIRE_CLASSES[c]?.color ?? '#888',
                            opacity: c === triage.classId ? 1 : 0.45,
                          }}
                        />
                      </div>
                      <span className="w-9 text-right tabular-nums">{Math.round(p * 100)}%</span>
                    </div>
                  );
                })}
              </div>
            </Section>

            <Section title="Composite risk">
              <div className="flex items-end gap-3">
                <span
                  className={cn(
                    'font-mono text-3xl font-black',
                    triage.risk.total >= 70 ? 'text-rose-400' : triage.risk.total >= 40 ? 'text-amber-400' : 'text-cyan-400',
                  )}
                >
                  {triage.risk.total}
                </span>
                <span className="pb-1 text-white/40">/ 100</span>
              </div>
              <div className="mt-2 grid grid-cols-4 gap-1 text-center text-[10px]">
                {(['severity', 'anomaly', 'spread', 'exposure'] as const).map((k) => (
                  <div key={k} className="rounded bg-white/5 p-1">
                    <div className="uppercase text-white/40">{k}</div>
                    <div className="font-mono">{(triage.risk[k] as number).toFixed(2)}</div>
                  </div>
                ))}
              </div>
            </Section>

            <Section title="Live Meteorology & Plume (Open-Meteo)">
              {loadingWeather ? (
                <div className="py-2 text-white/50 animate-pulse text-[11px]">
                  Connecting to Open-Meteo for atmospheric vectors…
                </div>
              ) : weather ? (
                <div className="space-y-2">
                  <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
                    <div className="rounded bg-white/5 p-2 flex items-center gap-2">
                      <Wind className="h-4 w-4 text-purple-400 shrink-0" />
                      <div>
                        <div className="text-[9px] uppercase text-white/40">Wind Speed</div>
                        <div className="font-bold text-purple-200">{weather.windSpeedMps.toFixed(1)} m/s</div>
                        <div className="text-[9px] text-white/40">Gusts: {weather.windGustsMps.toFixed(1)} m/s</div>
                      </div>
                    </div>
                    <div className="rounded bg-white/5 p-2 flex items-center gap-2">
                      <Compass className="h-4 w-4 text-cyan-400 shrink-0" />
                      <div>
                        <div className="text-[9px] uppercase text-white/40">Advection Bearing</div>
                        <div className="font-bold text-cyan-200">{weather.windDirectionDeg}° {weather.cardinal}</div>
                        <div className="text-[9px] text-white/40">Origin: {weather.windDirectionMetDeg}°</div>
                      </div>
                    </div>
                    <div className="rounded bg-white/5 p-2 flex items-center gap-2">
                      <Thermometer className="h-4 w-4 text-amber-400 shrink-0" />
                      <div>
                        <div className="text-[9px] uppercase text-white/40">Temperature</div>
                        <div className="font-bold text-amber-200">{weather.temperatureC.toFixed(1)} °C</div>
                      </div>
                    </div>
                    <div className="rounded bg-white/5 p-2 flex items-center gap-2">
                      <Droplets className="h-4 w-4 text-blue-400 shrink-0" />
                      <div>
                        <div className="text-[9px] uppercase text-white/40">Rel. Humidity</div>
                        <div className="font-bold text-blue-200">{weather.relativeHumidityPct}%</div>
                      </div>
                    </div>
                  </div>
                  <div className="text-[9px] text-white/40 flex justify-between items-center px-1">
                    <span>{weather.source}</span>
                    <span className="text-purple-300">50% &amp; 90% Plume Envelope</span>
                  </div>
                </div>
              ) : (
                <div className="text-[11px] text-white/50">
                  Using default meteorological baseline (6.5 m/s @ 135° SE).
                </div>
              )}
            </Section>

            <Section title="Evidence">
              <ul className="space-y-1.5">
                {triage.explanation.map((line, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="mt-0.5 text-orange-400">▹</span>
                    <span className="text-white/80">{line}</span>
                  </li>
                ))}
              </ul>
            </Section>

            <Section title="Detection">
              <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-[11px]">
                <Field k="FRP" v={`${triage.detection.frp.toFixed(1)} MW`} />
                <Field k="Anomaly" v={triage.anomalyScore.toFixed(2)} />
                <Field k="Brightness" v={`${triage.detection.brightness.toFixed(0)} K`} />
                <Field k="Confidence" v={triage.detection.confidenceLevel} />
                <Field k="Facility z" v={triage.context.facilityZ ? `${triage.context.facilityZ}σ` : 'n/a'} />
                <Field k="Biome" v={triage.context.biome} />
              </dl>
            </Section>

            <p className="rounded border border-white/10 bg-white/5 p-2 text-[10px] leading-relaxed text-white/45">
              {triage.disclaimer}
            </p>

            {/* CRITICAL alert payload card — shown when route is CRITICAL */}
            {triage.routeState === 'CRITICAL' && (
              <div className="rounded border border-rose-500/60 bg-rose-950/40 p-3 space-y-1.5 animate-pulse-glow-rose">
                <div className="text-[10px] font-mono font-black uppercase tracking-widest text-rose-400 flex items-center gap-1.5">
                  <ShieldAlert className="h-3.5 w-3.5" />
                  OUTBOUND ALERT — PHYSICAL RELAY TRIGGERED
                </div>
                <div className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-[11px]">
                  <div className="flex justify-between col-span-2">
                    <span className="text-white/40">Facility</span>
                    <span className="text-rose-200 font-bold truncate ml-2">
                      {triage.context.industrialPolygon?.name ?? triage.context.nearestFacility?.name ?? 'Unknown'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/40">Lat</span>
                    <span className="text-white/80">{triage.detection.latitude.toFixed(4)}°</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/40">Lon</span>
                    <span className="text-white/80">{triage.detection.longitude.toFixed(4)}°</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/40">FRP</span>
                    <span className="text-rose-300 font-bold">{triage.detection.frp.toFixed(1)} MW</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/40">Risk Score</span>
                    <span className="text-rose-300 font-bold">{triage.risk.total}/100</span>
                  </div>
                  <div className="flex justify-between col-span-2">
                    <span className="text-white/40">Acquired</span>
                    <span className="text-white/80">{triage.detection.acquiredAt.toISOString().replace('T', ' ').slice(0, 19)} UTC</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </aside>
    </>
  );
}

function RouteBanner({ route }: { route: RouteState }) {
  const s = ROUTE_STYLE[route];
  return (
    <div className={cn('flex items-center gap-2 border p-2.5 text-sm font-bold uppercase tracking-wide', s.bg)}>
      <s.icon className="h-4 w-4" />
      Route: {s.label}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded border border-white/10 bg-white/[0.03] p-3">
      <h3 className="mb-2 text-[10px] font-bold uppercase tracking-widest text-white/40">{title}</h3>
      {children}
    </section>
  );
}

function Field({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between">
      <dt className="text-white/40">{k}</dt>
      <dd className="text-white/85">{v}</dd>
    </div>
  );
}
