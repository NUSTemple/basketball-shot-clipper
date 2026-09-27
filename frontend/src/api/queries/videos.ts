import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api } from '../client'
import type { Video } from '../types'

export const videoKeys = {
  list: ['videos'] as const,
  detail: (videoId: number) => ['videos', videoId] as const,
}

export function useVideos() {
  return useQuery({
    queryKey: videoKeys.list,
    queryFn: () => api.get<Video[]>('/api/v2/videos'),
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

export function useRegisterVideo() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: { gcs_relpath: string; original_filename: string; calibration_profile_id?: number }) =>
      api.post<Video>('/api/v2/videos', body),
    onSuccess: () => qc.invalidateQueries({ queryKey: videoKeys.list }),
  })
}

export function useAttachCalibrationProfile(videoId: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (calibration_profile_id: number) =>
      api.patch<Video>(`/api/v2/videos/${videoId}`, { calibration_profile_id }),
    onSuccess: (video) => {
      qc.setQueryData(videoKeys.detail(videoId), video)
      qc.invalidateQueries({ queryKey: videoKeys.list })
    },
  })
}
