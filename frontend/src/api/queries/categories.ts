import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api } from '../client'
import type { LabelCategory } from '../types'

const activeKey = ['label-categories'] as const
const allKey = ['label-categories', 'all'] as const

export function useLabelCategories() {
  return useQuery({
    queryKey: activeKey,
    queryFn: () => api.get<LabelCategory[]>('/api/v2/label-categories'),
  })
}

// Includes soft-deleted (active=false) categories - only meaningful for the
// admin management screen, which shows every row regardless of active state.
export function useAllLabelCategories() {
  return useQuery({
    queryKey: allKey,
    queryFn: () => api.get<LabelCategory[]>('/api/v2/label-categories?all=1'),
  })
}

function invalidateCategoryQueries(qc: ReturnType<typeof useQueryClient>) {
  qc.invalidateQueries({ queryKey: activeKey })
  qc.invalidateQueries({ queryKey: allKey })
}

export function useCreateLabelCategory() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => api.post<LabelCategory>('/api/v2/label-categories', { name }),
    onSuccess: () => invalidateCategoryQueries(qc),
  })
}

export function useUpdateLabelCategory() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, ...patch }: { id: number; name?: string; active?: boolean }) =>
      api.patch<LabelCategory>(`/api/v2/label-categories/${id}`, patch),
    onSuccess: () => invalidateCategoryQueries(qc),
  })
}
