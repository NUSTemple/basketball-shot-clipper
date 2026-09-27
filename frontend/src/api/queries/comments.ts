import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api } from '../client'
import type { Comment } from '../types'

export const commentKeys = {
  list: (markerId: number) => ['markers', markerId, 'comments'] as const,
}

export function useComments(markerId: number | null) {
  return useQuery({
    queryKey: commentKeys.list(markerId ?? -1),
    queryFn: () => api.get<Comment[]>(`/api/v2/markers/${markerId}/comments`),
    enabled: markerId != null,
    refetchInterval: 5000,
  })
}

export function useCreateComment(markerId: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (text: string) => api.post<Comment>(`/api/v2/markers/${markerId}/comments`, { text }),
    onSuccess: () => qc.invalidateQueries({ queryKey: commentKeys.list(markerId) }),
  })
}

export function useDeleteComment(markerId: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (commentId: number) => api.del(`/api/v2/comments/${commentId}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: commentKeys.list(markerId) }),
  })
}
