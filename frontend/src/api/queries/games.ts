import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api } from '../client'
import type { Game } from '../types'

export const gameKeys = {
  list: ['games'] as const,
  detail: (gameId: number) => ['games', gameId] as const,
}

export function useGames() {
  return useQuery({
    queryKey: gameKeys.list,
    queryFn: () => api.get<Game[]>('/api/v2/games'),
  })
}

export function useGame(gameId: number | null | undefined) {
  return useQuery({
    queryKey: gameKeys.detail(gameId ?? -1),
    queryFn: () => api.get<Game>(`/api/v2/games/${gameId}`),
    enabled: gameId != null,
  })
}

export function useCreateGame() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: { location: string; game_date: string; name?: string }) =>
      api.post<Game>('/api/v2/games', body),
    onSuccess: () => qc.invalidateQueries({ queryKey: gameKeys.list }),
  })
}

export function useAddGamePlayer(gameId: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => api.post<Game>(`/api/v2/games/${gameId}/players`, { name }),
    onSuccess: (game) => {
      qc.setQueryData(gameKeys.detail(gameId), game)
      qc.invalidateQueries({ queryKey: gameKeys.list })
    },
  })
}

export function useRemoveGamePlayer(gameId: number) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (rosterPlayerId: number) => api.del(`/api/v2/games/${gameId}/players/${rosterPlayerId}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: gameKeys.detail(gameId) })
      qc.invalidateQueries({ queryKey: gameKeys.list })
    },
  })
}
