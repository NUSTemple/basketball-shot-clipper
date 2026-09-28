import { useState } from 'react'

import {
  calibrationFrameUrl,
  useCalibrationProfiles,
  useCreateCalibrationProfile,
} from '../api/queries/calibrationProfiles'
import { useVideos } from '../api/queries/videos'
import { HoopCalibrationCanvas } from '../components/calibration/HoopCalibrationCanvas'

export function CalibrationProfilesPage() {
  const { data: profiles } = useCalibrationProfiles()
  const { data: videos } = useVideos()
  const createProfile = useCreateCalibrationProfile()

  const [sourceVideoId, setSourceVideoId] = useState<number | ''>('')
  const [name, setName] = useState('')
  const [box, setBox] = useState<{ bbox: [number, number, number, number]; w: number; h: number } | null>(null)

  const save = (e: React.FormEvent) => {
    e.preventDefault()
    if (!box || !name.trim()) return
    createProfile.mutate(
      { name: name.trim(), hoop_bbox_norm: box.bbox, frame_width: box.w, frame_height: box.h },
      {
        onSuccess: () => {
          setName('')
          setBox(null)
          setSourceVideoId('')
        },
      },
    )
  }

  return (
    <div className="mx-auto max-w-3xl space-y-8 p-6">
      <div>
        <h1 className="mb-1 text-xl font-semibold">Basket Calibrations</h1>
        <p className="mb-3 text-sm text-slate-500">
          Where the basket sits in the frame - auto-detection only looks there. Usually drawn per video
          (from the video's own page); save one here only for a camera position you'll reuse.
        </p>
        <ul className="space-y-1">
          {profiles?.map((p) => (
            <li key={p.id} className="rounded-md bg-white px-3 py-2 text-sm shadow-sm">
              <span className="font-medium">{p.name}</span>
              <span className="ml-2 text-slate-400">
                bbox [{p.hoop_bbox_norm.map((v) => v.toFixed(2)).join(', ')}]
              </span>
            </li>
          ))}
          {profiles?.length === 0 && <p className="text-sm text-slate-500">No saved basket calibrations yet.</p>}
        </ul>
      </div>

      <div>
        <h2 className="mb-3 text-lg font-semibold">New basket calibration</h2>
        <label className="mb-1 block text-sm font-medium text-slate-700">
          Draw the box on a frame from…
        </label>
        <select
          value={sourceVideoId}
          onChange={(e) => {
            setSourceVideoId(e.target.value ? Number(e.target.value) : '')
            setBox(null)
          }}
          className="mb-3 w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm"
        >
          <option value="">Select a video…</option>
          {videos?.map((v) => (
            <option key={v.id} value={v.id}>
              {v.original_filename}
            </option>
          ))}
        </select>

        {sourceVideoId && (
          <HoopCalibrationCanvas
            imageSrc={calibrationFrameUrl(sourceVideoId)}
            onBoxChange={(bbox, w, h) => setBox({ bbox, w, h })}
          />
        )}

        <form onSubmit={save} className="mt-4 flex gap-2">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Name, e.g. Home Court, Camera 1"
            className="flex-1 rounded-md border border-slate-300 px-2 py-1.5 text-sm"
          />
          <button
            type="submit"
            disabled={!box || !name.trim() || createProfile.isPending}
            className="rounded-md bg-orange-500 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
          >
            Save calibration
          </button>
        </form>
        {createProfile.isError && (
          <p className="mt-2 text-sm text-red-600">{(createProfile.error as Error).message}</p>
        )}
      </div>
    </div>
  )
}
