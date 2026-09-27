import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api } from '../client'
import type { Profile } from '../types'

const profileKey = ['profile'] as const

export function useCurrentUser() {
  return useQuery({
    queryKey: profileKey,
    queryFn: () => api.get<Profile>('/api/v2/profile'),
    staleTime: Infinity, // identity doesn't change mid-session
  })
}

export function useUpdateProfile() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (patch: { display_name?: string; avatar_url?: string }) =>
      api.patch<Profile>('/api/v2/profile', patch),
    onSuccess: (profile) => qc.setQueryData(profileKey, profile),
  })
}
