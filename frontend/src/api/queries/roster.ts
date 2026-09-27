import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api } from '../client'

const rosterKey = ['roster'] as const

export function useRoster() {
  return useQuery({
    queryKey: rosterKey,
    queryFn: () => api.get<string[]>('/api/v2/roster'),
  })
}

export function useAddRosterPlayer() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => api.post<string[]>('/api/v2/roster', { name }),
    onSuccess: (players) => qc.setQueryData(rosterKey, players),
  })
}
