import { useMutation, useQueryClient } from '@tanstack/react-query'

import { api } from '../client'
import type { Video } from '../types'
import { videoKeys } from './videos'

export type UploadStage = 'requesting-url' | 'uploading' | 'confirming' | 'registering'

interface CreateUploadResponse {
  upload_url: string
  path: string
  blob_name: string
  content_type: string
  duplicate: { path: string; size_display: string } | null
}

export interface UploadVideoArgs {
  file: File
  calibrationProfileId?: number
  onStage?: (stage: UploadStage) => void
}

// Composes the existing /api/uploads (mint a GCS signed PUT URL) +
// /api/uploads/complete (server-side storage-cap enforcement hook) flow,
// unchanged from the legacy UI, with the new v2 step of registering the
// result as a `videos` row. blob_name (added to /api/uploads' response as
// a Phase 5 prerequisite fix) is exactly Video.gcs_relpath.
async function uploadVideo({ file, calibrationProfileId, onStage }: UploadVideoArgs): Promise<Video> {
  onStage?.('requesting-url')
  const created = await api.post<CreateUploadResponse>('/api/uploads', {
    filename: file.name,
    content_type: file.type || 'video/mp4',
    size: file.size,
  })

  onStage?.('uploading')
  const putRes = await fetch(created.upload_url, {
    method: 'PUT',
    headers: { 'Content-Type': created.content_type },
    body: file,
  })
  if (!putRes.ok) {
    throw new Error(`upload to storage failed (${putRes.status})`)
  }

  onStage?.('confirming')
  await api.post('/api/uploads/complete')

  onStage?.('registering')
  return api.post<Video>('/api/v2/videos', {
    gcs_relpath: created.blob_name,
    original_filename: file.name,
    calibration_profile_id: calibrationProfileId,
  })
}

export function useUploadVideo() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: uploadVideo,
    onSuccess: () => qc.invalidateQueries({ queryKey: videoKeys.list }),
  })
}
