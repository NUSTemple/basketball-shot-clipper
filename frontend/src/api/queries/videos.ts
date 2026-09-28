import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api } from '../client'
import type { Video } from '../types'

export const videoKeys = {
  all: ['videos'] as const, // prefix - invalidate this to hit every list, scoped or not
  list: (gameId?: number) => ['videos', { gameId: gameId ?? null }] as const,
  detail: (videoId: number) => ['videos', videoId] as const,
}

export function useVideos(gameId?: number) {
  return useQuery({
    queryKey: videoKeys.list(gameId),
    queryFn: () => api.get<Video[]>(gameId ? `/api/v2/videos?game_id=${gameId}` : '/api/v2/videos'),
    refetchInterval: 5000, // picks up other users' uploads and status changes
  })
}

export function useVideo(videoId: number) {
  return useQuery({
    queryKey: videoKeys.detail(videoId),
    queryFn: () => api.get<Video>(`/api/v2/videos/${videoId}`),
    refetchInterval: (query) => (query.state.data?.status === 'detecting' ? 3000 : false),
  })
}

export function useAttachCalibrationProfile(videoId: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (calibration_profile_id: number) =>
      api.patch<Video>(`/api/v2/videos/${videoId}`, { calibration_profile_id }),
    onSuccess: (video) => {
      qc.setQueryData(videoKeys.detail(videoId), video)
      qc.invalidateQueries({ queryKey: videoKeys.all })
    },
  })
}
