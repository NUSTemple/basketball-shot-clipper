import { useCreateMarker } from '../../api/queries/markers'
import type { Marker } from '../../api/types'

interface MarkerNavProps {
  videoId: number
  markers: Marker[]
  currentTime: number
  selectedMarkerId: number | null
  onSelectMarker: (markerId: number) => void
  onSeek: (seconds: number) => void
}

export function MarkerNav({
  videoId,
  markers,
  currentTime,
  selectedMarkerId,
  onSelectMarker,
  onSeek,
}: MarkerNavProps) {
  const createMarker = useCreateMarker(videoId)
  const sorted = [...markers].sort((a, b) => a.timestamp_s - b.timestamp_s)

  const jump = (direction: 1 | -1) => {
    if (sorted.length === 0) return
    const currentIndex = sorted.findIndex((m) => m.id === selectedMarkerId)
    let nextIndex: number
    if (currentIndex === -1) {
      // no marker selected - jump to the next/previous one relative to playhead
      nextIndex =
        direction === 1
          ? sorted.findIndex((m) => m.timestamp_s > currentTime)
          : [...sorted].reverse().findIndex((m) => m.timestamp_s < currentTime)
      if (direction === -1 && nextIndex !== -1) nextIndex = sorted.length - 1 - nextIndex
    } else {
      nextIndex = currentIndex + direction
    }
    if (nextIndex < 0 || nextIndex >= sorted.length || nextIndex === -1) return
    const target = sorted[nextIndex]
    onSelectMarker(target.id)
    onSeek(target.timestamp_s)
  }

  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        onClick={() => jump(-1)}
        className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium hover:bg-slate-100"
      >
        ← Prev marker
      </button>
      <button
        type="button"
        onClick={() => jump(1)}
        className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium hover:bg-slate-100"
      >
        Next marker →
      </button>
      <button
        type="button"
        disabled={createMarker.isPending}
        onClick={() => createMarker.mutate(currentTime)}
        className="ml-auto rounded-md bg-orange-500 px-3 py-1.5 text-sm font-medium text-white hover:bg-orange-600 disabled:opacity-50"
      >
        + Add marker here
      </button>
    </div>
  )
}
