import { useMutation } from '@tanstack/react-query'

import { api } from '../client'
import type { ExportResult } from '../types'

export interface ExportFilter {
  categories?: string[]
  player?: string
  comment_keyword?: string
  pre?: number
  post?: number
}

export function useCreateExport() {
  return useMutation({
    mutationFn: (filter: ExportFilter) => api.post<ExportResult>('/api/v2/export', filter),
  })
}

export function exportDownloadUrl(jobId: string): string {
  return `/api/v2/export/${jobId}/download`
}
