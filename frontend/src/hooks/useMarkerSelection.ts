import { useSearchParams } from 'react-router-dom'

// "Which marker is open" lives in the URL (?marker=<id>), not useState/context -
// shareable, bookmarkable, and survives a refresh for free, reusing state
// react-router already owns rather than introducing a second source of truth.
export function useMarkerSelection(): [number | null, (markerId: number | null) => void] {
  const [searchParams, setSearchParams] = useSearchParams()
  const raw = searchParams.get('marker')
  const selected = raw != null && /^\d+$/.test(raw) ? Number(raw) : null

  const select = (markerId: number | null) => {
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        if (markerId == null) next.delete('marker')
        else next.set('marker', String(markerId))
        return next
      },
      { replace: true },
    )
  }

  return [selected, select]
}
