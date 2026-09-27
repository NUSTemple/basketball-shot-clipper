import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api } from '../client'
import type { Label } from '../types'

export const labelKeys = {
  list: (markerId: number) => ['markers', markerId, 'labels'] as const,
}

export function useLabels(markerId: number | null) {
  return useQuery({
    queryKey: labelKeys.list(markerId ?? -1),
    queryFn: () => api.get<Label[]>(`/api/v2/markers/${markerId}/labels`),
    enabled: markerId != null,
    refetchInterval: 5000,
  })
}

export function useCreateLabel(markerId: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: { category_id: number; player_name?: string }) =>
      api.post<Label>(`/api/v2/markers/${markerId}/labels`, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: labelKeys.list(markerId) }),
  })
}

export function useDeleteLabel(markerId: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (labelId: number) => api.del(`/api/v2/labels/${labelId}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: labelKeys.list(markerId) }),
  })
}
