import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api } from '../client'
import type { CalibrationProfile } from '../types'

const listKey = ['calibration-profiles'] as const

export function useCalibrationProfiles() {
  return useQuery({
    queryKey: listKey,
    queryFn: () => api.get<CalibrationProfile[]>('/api/v2/calibration-profiles'),
  })
}

export interface NewCalibrationProfile {
  name: string
  hoop_bbox_norm: [number, number, number, number]
  frame_width?: number
  frame_height?: number
}

export function useCreateCalibrationProfile() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: NewCalibrationProfile) =>
      api.post<CalibrationProfile>('/api/v2/calibration-profiles', body),
    onSuccess: () => qc.invalidateQueries({ queryKey: listKey }),
  })
}

export function calibrationFrameUrl(videoId: number, t?: number): string {
  return `/api/v2/videos/${videoId}/calibration-frame${t != null ? `?t=${t}` : ''}`
}

export function useUpdateCalibrationProfile(profileId: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: { hoop_bbox_norm: [number, number, number, number]; frame_width?: number; frame_height?: number }) =>
      api.patch<CalibrationProfile>(`/api/v2/calibration-profiles/${profileId}`, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: listKey }),
  })
}
