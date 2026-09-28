import { useMutation, useQueryClient } from '@tanstack/react-query'

import { api } from '../client'
import type { Video } from '../types'
import { videoKeys } from './videos'

export type UploadStage = 'requesting-url' | 'uploading' | 'confirming' | 'registering'

export interface UploadProgress {
  loaded: number
  total: number
  etaSeconds: number | null
}

interface CreateUploadResponse {
  upload_url: string
  path: string
  blob_name: string
  content_type: string
  duplicate: { path: string; size_display: string } | null
}

export interface UploadVideoArgs {
  file: File
  gameId: number
  calibrationProfileId?: number
  onStage?: (stage: UploadStage) => void
  onProgress?: (progress: UploadProgress) => void
}

// fetch() has no upload-progress event - XMLHttpRequest is the only widely
// supported way to observe bytes sent during a PUT, so the direct-to-GCS
// step uses it instead of fetch, unlike every other request in this app.
function putWithProgress(
  url: string,
  file: File,
  contentType: string,
  onProgress?: (progress: UploadProgress) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('PUT', url)
    xhr.setRequestHeader('Content-Type', contentType)

    const startedAt = Date.now()
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable || !onProgress) return
      const elapsedSec = (Date.now() - startedAt) / 1000
      const bytesPerSec = elapsedSec > 0 ? e.loaded / elapsedSec : 0
      const etaSeconds = bytesPerSec > 0 ? (e.total - e.loaded) / bytesPerSec : null
      onProgress({ loaded: e.loaded, total: e.total, etaSeconds })
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve()
      else reject(new Error(`upload to storage failed (${xhr.status})`))
    }
    xhr.onerror = () => reject(new Error('upload to storage failed (network error)'))
    xhr.send(file)
  })
}

// Composes the existing /api/uploads (mint a GCS signed PUT URL) +
// /api/uploads/complete (server-side storage-cap enforcement hook) flow,
// unchanged from the legacy UI, with the new v2 step of registering the
// result as a `videos` row. blob_name (added to /api/uploads' response as
// a Phase 5 prerequisite fix) is exactly Video.gcs_relpath. The basket
// calibration is chosen explicitly (or skipped and attached later) - each
// video's basket is usually in a different spot, so nothing is auto-reused.
async function uploadVideo({ file, gameId, calibrationProfileId, onStage, onProgress }: UploadVideoArgs): Promise<Video> {
  onStage?.('requesting-url')
  const created = await api.post<CreateUploadResponse>('/api/uploads', {
    filename: file.name,
    content_type: file.type || 'video/mp4',
    size: file.size,
  })

  onStage?.('uploading')
  await putWithProgress(created.upload_url, file, created.content_type, onProgress)

  onStage?.('confirming')
  await api.post('/api/uploads/complete')

  onStage?.('registering')
  return api.post<Video>('/api/v2/videos', {
    gcs_relpath: created.blob_name,
    original_filename: file.name,
    game_id: gameId,
    calibration_profile_id: calibrationProfileId,
  })
}

export function useUploadVideo() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: uploadVideo,
    onSuccess: () => qc.invalidateQueries({ queryKey: videoKeys.all }),
  })
}
