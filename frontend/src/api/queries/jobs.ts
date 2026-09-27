import { useQuery } from '@tanstack/react-query'

import { api } from '../client'
import { isTerminalJobState, type Job } from '../types'

// Shared by the "detect now" banner and the export flow - polls fast while
// a job is live and stops entirely once it reaches a terminal state, so an
// old finished job's tab left open doesn't keep polling forever.
export function useJob(jobId: string | null) {
  return useQuery({
    queryKey: ['jobs', jobId],
    queryFn: () => api.get<Job>(`/api/v2/jobs/${jobId}`),
    enabled: jobId != null,
    refetchInterval: (query) => {
      const state = query.state.data?.state
      return state && isTerminalJobState(state) ? false : 1500
    },
  })
}
