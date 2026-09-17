/**
 * "Alert nearby devices" for one triaged hotspot.
 *
 * The alert is always *centred on the fire* and carries its classification, so
 * what handsets receive matches what the model actually decided. The demo toggle
 * widens the broadcast radius until it reaches the venue rather than moving the
 * alert somewhere it isn't — a real CBC picks the footprint the same way, so the
 * geography stays honest either way.
 *
 * See docs/cell-broadcast.md.
 */

import { useEffect, useState } from 'react';
import { RadioTower, Loader2 } from 'lucide-react';
import { jvalyxApi, type BackendCbsReceipt } from '../../services/api';
import type { RouteState } from '../../types';
import { cn } from './ui';

const POLL_INTERVAL_MS = 4000;
const EARTH_RADIUS_KM = 6371;
/** Margin past the venue, so a handset just outside the ring still receives. */
const VENUE_MARGIN_KM = 5;

function haversineKm(aLat: number, aLon: number, bLat: number, bLon: number): number {
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const dLat = toRad(bLat - aLat);
  const dLon = toRad(bLon - aLon);
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(aLat)) * Math.cos(toRad(bLat)) * Math.sin(dLon / 2) ** 2;
  return 2 * EARTH_RADIUS_KM * Math.asin(Math.sqrt(h));
}

interface FireAlertBroadcastProps {
  latitude: number;
  longitude: number;
  /** Triage reports this as a plain number, so it is not narrowed to FireClassId. */
  classId: number;
  className: string;
  routeState: RouteState;
  /** Facility or zone name when the hotspot fell inside one. */
  placeName?: string;
}

export function FireAlertBroadcast({
  latitude,
  longitude,
  classId,
  className,
  routeState,
  placeName,
}: FireAlertBroadcastProps) {
  const [attached, setAttached] = useState<number | null>(null);
  const [venue, setVenue] = useState<{ latitude: number; longitude: number } | null>(null);
  const [coverVenue, setCoverVenue] = useState(true);
  const [sending, setSending] = useState(false);
  const [receipt, setReceipt] = useState<BackendCbsReceipt | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const status = await jvalyxApi.cbsStatus();
        if (cancelled) return;
        setAttached(status.devices_attached);
        setVenue(status.venue);
      } catch {
        if (!cancelled) setAttached(null);
      }
    };
    void poll();
    const timer = window.setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  // A new hotspot is a new alert; don't leave the last receipt sitting there.
  useEffect(() => {
    setReceipt(null);
    setError(null);
  }, [latitude, longitude]);

  const where = placeName ?? `${latitude.toFixed(3)}°, ${longitude.toFixed(3)}°`;
  const venueDistanceKm = venue ? haversineKm(latitude, longitude, venue.latitude, venue.longitude) : null;
  const radiusKm =
    coverVenue && venueDistanceKm !== null
      ? Math.max(10, Math.ceil(venueDistanceKm + VENUE_MARGIN_KM))
      : undefined;

  const broadcast = async () => {
    setSending(true);
    setError(null);
    try {
      setReceipt(
        await jvalyxApi.cbsBroadcast({
          route_state: routeState,
          latitude,
          longitude,
          area_desc: where,
          event: `Class ${classId} — ${className}`,
          headline: `${routeState} · Class ${classId} ${className} detected at ${where}`,
          radius_km: radiusKm,
        }),
      );
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'broadcast failed');
    } finally {
      setSending(false);
    }
  };

  const unreachable = attached === null;
  const noHandsets = attached === 0;

  return (
    <div className="rounded border border-rose-500/40 bg-rose-950/30 p-2.5">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-rose-300">
          <RadioTower className="h-3.5 w-3.5 shrink-0" /> Public warning
        </span>
        <span className="font-mono text-[10px] text-white/45">
          {unreachable ? 'backend offline' : `${attached} handset${attached === 1 ? '' : 's'} attached`}
        </span>
      </div>

      <p className="mt-1.5 text-[11px] leading-snug text-white/55">
        Broadcasts a CAP alert naming this fire as <span className="text-white/85">Class {classId} — {className}</span>,
        centred on {where}.
      </p>

      <button
        type="button"
        onClick={broadcast}
        disabled={sending || unreachable || noHandsets}
        title={
          unreachable
            ? 'Backend unreachable'
            : noHandsets
              ? 'No handsets attached — open /cbs/console and scan the QR'
              : 'Broadcast this alert to attached handsets'
        }
        className={cn(
          'mt-2 flex w-full items-center justify-center gap-2 rounded py-2 text-[11px] font-black uppercase tracking-wider transition-colors',
          'bg-rose-600 text-white hover:bg-rose-500',
          'disabled:cursor-not-allowed disabled:bg-white/10 disabled:text-white/35',
        )}
      >
        {sending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RadioTower className="h-3.5 w-3.5" />}
        Alert nearby devices
      </button>

      <label className="mt-2 flex cursor-pointer items-start gap-2 text-[10px] leading-snug text-white/50">
        <input
          type="checkbox"
          checked={coverVenue}
          onChange={(e) => setCoverVenue(e.target.checked)}
          className="mt-0.5 shrink-0 accent-rose-500"
        />
        <span>
          Widen footprint to reach this room
          {venueDistanceKm !== null && (
            <span className="font-mono text-white/35"> · {radiusKm ?? Math.round(venueDistanceKm)} km</span>
          )}
          <span className="block text-white/30">
            Alert stays centred on the fire; only the radius changes.
          </span>
        </span>
      </label>

      {receipt && (
        <div className="mt-2 rounded bg-black/40 p-2 font-mono text-[10px] leading-relaxed text-emerald-300">
          <div>
            {receipt.devices_in_footprint}/{receipt.devices_attached} handsets · {receipt.cells_in_footprint.length} cell(s)
          </div>
          <div className="text-emerald-400/70">
            {receipt.cmas_class} · msg id {receipt.message_identifier} · {receipt.status}
          </div>
          {receipt.devices_in_footprint === 0 && (
            <div className="mt-1 text-amber-300">
              No handset is inside the footprint — tick the box above to widen it.
            </div>
          )}
        </div>
      )}

      {error && <div className="mt-2 font-mono text-[10px] text-rose-300">{error}</div>}
    </div>
  );
}
