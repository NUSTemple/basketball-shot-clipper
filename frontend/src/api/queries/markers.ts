import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api } from '../client'
import type { Marker, MarkerState } from '../types'

export const markerKeys = {
  list: (videoId: number) => ['videos', videoId, 'markers'] as const,
}

export function useMarkers(videoId: number) {
  return useQuery({
    queryKey: markerKeys.list(videoId),
    queryFn: () => api.get<Marker[]>(`/api/v2/videos/${videoId}/markers`),
    // catches auto-detect's results landing and other users' confirm/dismiss
    // edits, since visibility is fully open - see docs/REQUIREMENTS_V2.md #2
    refetchInterval: 5000,
  })
}

export function useCreateMarker(videoId: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (timestamp_s: number) =>
      api.post<Marker>(`/api/v2/videos/${videoId}/markers`, { timestamp_s }),
    onSuccess: () => qc.invalidateQueries({ queryKey: markerKeys.list(videoId) }),
  })
}

export function useUpdateMarkerState(videoId: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ markerId, state }: { markerId: number; state: MarkerState }) =>
      api.patch<Marker>(`/api/v2/markers/${markerId}`, { state }),
    onSuccess: () => qc.invalidateQueries({ queryKey: markerKeys.list(videoId) }),
  })
}

export function useRetimeMarker(videoId: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ markerId, timestamp_s }: { markerId: number; timestamp_s: number }) =>
      api.patch<Marker>(`/api/v2/markers/${markerId}`, { timestamp_s }),
    onSuccess: () => qc.invalidateQueries({ queryKey: markerKeys.list(videoId) }),
  })
}

export function useDeleteMarker(videoId: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (markerId: number) => api.del(`/api/v2/markers/${markerId}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: markerKeys.list(videoId) }),
  })
}
