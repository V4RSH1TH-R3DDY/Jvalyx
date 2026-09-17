/**
 * Optical satellite crop of a hotspot's location.
 *
 * Pulled from Esri World Imagery — the same source as the map's satellite
 * basemap, so no extra key and no new provider.
 *
 * IMPORTANT, and stated on the card itself: this is an *archival* optical
 * basemap, not the thermal acquisition that produced the detection. VIIRS/MODIS
 * detect fire in the infrared at 375 m–1 km per pixel; this imagery is
 * high-resolution visible light captured on some earlier date. It shows what is
 * *at* the location, never the fire itself. Labelling it otherwise would be a
 * claim the data does not support.
 */

import { useEffect, useState } from 'react';
import { Satellite, ImageOff, Loader2 } from 'lucide-react';
import { cn } from './ui';

const EXPORT_ENDPOINT =
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export';
const KM_PER_DEGREE_LAT = 111.32;
const IMAGE_PX = 360;

/** Ground width of the crop. Named for what an operator would ask for. */
const SPANS = [
  { label: 'Site', km: 1 },
  { label: 'Perimeter', km: 3 },
  { label: 'Area', km: 10 },
] as const;

function cropUrl(latitude: number, longitude: number, spanKm: number): string {
  const halfLat = spanKm / 2 / KM_PER_DEGREE_LAT;
  // Longitude degrees shrink toward the poles, so scale by cos(lat) to keep the
  // footprint square on the ground.
  const halfLon = halfLat / Math.max(Math.cos((latitude * Math.PI) / 180), 0.01);
  const bbox = [
    longitude - halfLon,
    latitude - halfLat,
    longitude + halfLon,
    latitude + halfLat,
  ].join(',');

  const params = new URLSearchParams({
    bbox,
    bboxSR: '4326',
    imageSR: '3857',
    size: `${IMAGE_PX},${IMAGE_PX}`,
    format: 'jpg',
    f: 'image',
  });
  return `${EXPORT_ENDPOINT}?${params.toString()}`;
}

interface SatelliteCropProps {
  latitude: number;
  longitude: number;
}

export function SatelliteCrop({ latitude, longitude }: SatelliteCropProps) {
  const [spanKm, setSpanKm] = useState<number>(SPANS[1].km);
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading');

  const url = cropUrl(latitude, longitude, spanKm);

  // A new hotspot or span means a new request; show the spinner again.
  useEffect(() => setState('loading'), [url]);

  return (
    <div className="rounded border border-white/15 bg-black/30">
      <div className="flex items-center justify-between border-b border-white/10 px-2.5 py-1.5">
        <span className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-white/70">
          <Satellite className="h-3.5 w-3.5 shrink-0" /> Site imagery
        </span>
        <div className="flex items-center gap-0.5">
          {SPANS.map((span) => (
            <button
              key={span.km}
              type="button"
              onClick={() => setSpanKm(span.km)}
              className={cn(
                'rounded px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide transition-colors',
                spanKm === span.km
                  ? 'bg-white/20 text-white'
                  : 'text-white/40 hover:bg-white/10 hover:text-white/70',
              )}
            >
              {span.label}
            </button>
          ))}
        </div>
      </div>

      <div className="relative aspect-square w-full overflow-hidden bg-[#0b0f14]">
        {state === 'loading' && (
          <div className="absolute inset-0 flex items-center justify-center text-white/40">
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
        )}

        {state === 'error' ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-1.5 px-4 text-center text-white/40">
            <ImageOff className="h-5 w-5" />
            <span className="text-[10px] leading-snug">
              Imagery unavailable — the tile service is unreachable.
            </span>
          </div>
        ) : (
          <img
            key={url}
            src={url}
            alt={`Satellite view of ${latitude.toFixed(4)}°, ${longitude.toFixed(4)}°`}
            width={IMAGE_PX}
            height={IMAGE_PX}
            loading="lazy"
            onLoad={() => setState('ready')}
            onError={() => setState('error')}
            className={cn(
              'h-full w-full object-cover transition-opacity duration-300',
              state === 'ready' ? 'opacity-100' : 'opacity-0',
            )}
          />
        )}

        {/* Crosshair on the detection's exact coordinates, which sit dead centre. */}
        {state === 'ready' && (
          <div className="pointer-events-none absolute inset-0">
            <div className="absolute left-1/2 top-1/2 h-px w-7 -translate-x-1/2 -translate-y-1/2 bg-rose-500/80" />
            <div className="absolute left-1/2 top-1/2 h-7 w-px -translate-x-1/2 -translate-y-1/2 bg-rose-500/80" />
            <div className="absolute left-1/2 top-1/2 h-11 w-11 -translate-x-1/2 -translate-y-1/2 rounded-full border border-rose-500/70" />
            <span className="absolute bottom-1 right-1.5 font-mono text-[9px] text-white/70 [text-shadow:0_1px_2px_rgba(0,0,0,.9)]">
              {spanKm} km across
            </span>
          </div>
        )}
      </div>

      <p className="px-2.5 py-1.5 text-[9px] leading-snug text-white/35">
        Archival optical basemap (Esri, Maxar, Earthstar Geographics) — site context only.
        The detection itself is thermal infrared from VIIRS/MODIS, not this imagery.
      </p>
    </div>
  );
}
