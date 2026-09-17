/**
 * Cell-broadcast trigger for the header.
 *
 * Fires a CAP alert at the venue's coordinates, which is where handsets that
 * declined GPS are placed — so the button reaches every attached phone in the
 * room. Event-driven broadcasts (targeting a real detection's coordinates) live
 * on the presenter console at `/cbs/console`; see `docs/cell-broadcast.md`.
 *
 * Self-contained on purpose: it owns its own polling and state so the header
 * does not have to thread props for a demo affordance.
 */

import React from 'react';
import { RadioTower, Loader2 } from 'lucide-react';
import { jvalyxApi, type BackendCbsReceipt } from '../services/api';

const POLL_INTERVAL_MS = 3000;
/** How long the receipt stays on screen before the button returns to idle. */
const RECEIPT_LINGER_MS = 6000;

interface CellBroadcastButtonProps {
  /** Shown as the CAP area description. Defaults to the venue. */
  areaDesc?: string;
}

export const CellBroadcastButton: React.FC<CellBroadcastButtonProps> = ({
  areaDesc = 'Venue — live demonstration',
}) => {
  const [attached, setAttached] = React.useState<number | null>(null);
  const [venue, setVenue] = React.useState<{ latitude: number; longitude: number } | null>(null);
  const [sending, setSending] = React.useState(false);
  const [receipt, setReceipt] = React.useState<BackendCbsReceipt | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const status = await jvalyxApi.cbsStatus();
        if (cancelled) return;
        setAttached(status.devices_attached);
        setVenue(status.venue);
      } catch {
        // Backend offline: the dashboard still runs, the button just idles.
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

  React.useEffect(() => {
    if (!receipt) return;
    const timer = window.setTimeout(() => setReceipt(null), RECEIPT_LINGER_MS);
    return () => window.clearTimeout(timer);
  }, [receipt]);

  const broadcast = async () => {
    if (!venue) return;
    setSending(true);
    setError(null);
    try {
      setReceipt(
        await jvalyxApi.cbsBroadcast({
          route_state: 'CRITICAL',
          latitude: venue.latitude,
          longitude: venue.longitude,
          area_desc: areaDesc,
          radius_km: 10,
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
    <div className="flex items-center gap-2">
      {receipt && (
        <span className="font-mono text-[11px] text-emerald-400">
          {receipt.devices_in_footprint}/{receipt.devices_attached} handsets · {receipt.cmas_class} ·{' '}
          {receipt.message_identifier}
        </span>
      )}
      {error && <span className="font-mono text-[11px] text-rose-400">{error}</span>}

      <button
        onClick={broadcast}
        disabled={sending || unreachable || noHandsets}
        title={
          unreachable
            ? 'Backend unreachable'
            : noHandsets
              ? 'No handsets attached — open /cbs/console and scan the QR'
              : `Broadcast a CAP alert to ${attached} attached handset(s)`
        }
        className="flex items-center gap-1.5 border border-rose-800/80 bg-rose-950/40 px-2.5 py-1.5 font-mono text-xs font-bold tracking-wider text-rose-300 transition-colors hover:border-rose-600 hover:bg-rose-900/50 hover:text-rose-100 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {sending ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : (
          <RadioTower className="h-3.5 w-3.5" />
        )}
        SIMULATE ALERT
        {!unreachable && (
          <span className="ml-0.5 rounded-sm bg-rose-900/70 px-1 font-tabular text-[10px] text-rose-200">
            {attached}
          </span>
        )}
      </button>
    </div>
  );
};
