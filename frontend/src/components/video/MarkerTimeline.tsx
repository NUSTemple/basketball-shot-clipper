import type { Marker } from '../../api/types'
import { formatTime } from '../../lib/time'

interface MarkerTimelineProps {
  duration: number
  currentTime: number
  markers: Marker[]
  selectedMarkerId: number | null
  onSeek: (seconds: number) => void
  onSelectMarker: (markerId: number) => void
}

const STATE_DOT: Record<Marker['state'], string> = {
  unconfirmed: 'bg-amber-500',
  confirmed: 'bg-green-500',
  dismissed: 'bg-slate-400',
}

export function MarkerTimeline({
  duration,
  currentTime,
  markers,
  selectedMarkerId,
  onSeek,
  onSelectMarker,
}: MarkerTimelineProps) {
  if (!duration) return null

  const pct = (seconds: number) => `${Math.min(100, Math.max(0, (seconds / duration) * 100))}%`

  // Clicking the bare track seeks (this is a scrub bar); dropping a new
  // manual marker is a separate explicit action (see MarkerNav's "Add
  // marker here" button) rather than overloading the same click, which
  // would make every seek ambiguous with marker creation.
  const handleTrackClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const fraction = (e.clientX - rect.left) / rect.width
    onSeek(fraction * duration)
  }

  return (
    <div className="select-none">
      <div
        className="relative h-10 cursor-pointer rounded-md bg-slate-200"
        onClick={handleTrackClick}
      >
        <div
          className="pointer-events-none absolute top-0 h-full w-0.5 bg-slate-900"
          style={{ left: pct(currentTime) }}
        />
        {markers.map((marker) => (
          <button
            key={marker.id}
            type="button"
            title={`${formatTime(marker.timestamp_s)} · ${marker.source} · ${marker.state}`}
            onClick={(e) => {
              e.stopPropagation()
              onSelectMarker(marker.id)
            }}
            className={`absolute top-1/2 size-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 ${STATE_DOT[marker.state]} ${
              marker.id === selectedMarkerId ? 'border-slate-900 ring-2 ring-slate-900' : 'border-white'
            } ${marker.source === 'manual' ? 'rotate-45 rounded-none' : ''}`}
            style={{ left: pct(marker.timestamp_s) }}
          />
        ))}
      </div>
      <div className="mt-1 flex justify-between text-xs text-slate-500">
        <span>{formatTime(currentTime)}</span>
        <span>{formatTime(duration)}</span>
      </div>
    </div>
  )
}
