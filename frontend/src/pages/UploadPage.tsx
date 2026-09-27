import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { useCalibrationProfiles } from '../api/queries/calibrationProfiles'
import { type UploadStage, useUploadVideo } from '../api/queries/uploads'
import { UploadDropzone } from '../components/upload/UploadDropzone'

const STAGE_LABEL: Record<UploadStage, string> = {
  'requesting-url': 'Requesting upload slot…',
  uploading: 'Uploading to storage…',
  confirming: 'Confirming upload…',
  registering: 'Registering video…',
}

export function UploadPage() {
  const navigate = useNavigate()
  const { data: profiles } = useCalibrationProfiles()
  const upload = useUploadVideo()
  const [calibrationProfileId, setCalibrationProfileId] = useState<number | ''>('')
  const [stage, setStage] = useState<UploadStage | null>(null)

  const handleFile = (file: File) => {
    upload.mutate(
      {
        file,
        calibrationProfileId: calibrationProfileId || undefined,
        onStage: setStage,
      },
      {
        onSuccess: (video) => navigate(`/videos/${video.id}`),
      },
    )
  }

  return (
    <div className="mx-auto max-w-xl p-6">
      <h1 className="mb-4 text-xl font-semibold">Upload a video</h1>

      <div className="mb-4">
        <label className="mb-1 block text-sm font-medium text-slate-700">
          Calibration profile (optional - attach one later if you're not sure)
        </label>
        <select
          value={calibrationProfileId}
          onChange={(e) => setCalibrationProfileId(e.target.value ? Number(e.target.value) : '')}
          disabled={upload.isPending}
          className="w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm"
        >
          <option value="">No profile yet</option>
          {profiles?.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      <UploadDropzone onFileSelected={handleFile} disabled={upload.isPending} />

      {stage && upload.isPending && (
        <p className="mt-3 text-sm text-slate-600">{STAGE_LABEL[stage]}</p>
      )}
      {upload.isError && (
        <p className="mt-3 text-sm text-red-600">{(upload.error as Error).message}</p>
      )}
    </div>
  )
}
