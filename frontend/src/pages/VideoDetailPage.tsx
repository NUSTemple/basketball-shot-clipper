import { useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import {
  calibrationFrameUrl,
  useCalibrationProfiles,
  useCreateCalibrationProfile,
} from '../api/queries/calibrationProfiles'
import { useGame, useGames } from '../api/queries/games'
import { useCurrentUser } from '../api/queries/profile'
import { useAttachCalibrationProfile, useReassignGame, useVideo } from '../api/queries/videos'
import { useMarkers } from '../api/queries/markers'
import type { Video } from '../api/types'
import { HoopCalibrationCanvas } from '../components/calibration/HoopCalibrationCanvas'
import { JobStatusBanner } from '../components/jobs/JobStatusBanner'
import { MarkerDetailPanel } from '../components/video/MarkerDetailPanel'
import { MarkerNav } from '../components/video/MarkerNav'
import { MarkerTimeline } from '../components/video/MarkerTimeline'
import { VideoPlayer, type VideoPlayerHandle } from '../components/video/VideoPlayer'
import { useMarkerSelection } from '../hooks/useMarkerSelection'

export function VideoDetailPage() {
  const { videoId: videoIdParam } = useParams<{ videoId: string }>()
  const videoId = Number(videoIdParam)

  const { data: video } = useVideo(videoId)
  const { data: markers } = useMarkers(videoId)
  const { data: currentUser } = useCurrentUser()
  const { data: game } = useGame(video?.game_id)
  const [selectedMarkerId, selectMarker] = useMarkerSelection()

  const playerRef = useRef<VideoPlayerHandle>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [pendingJobId, setPendingJobId] = useState<string | null>(null)

  if (!video) return <div className="p-6 text-slate-500">Loading…</div>

  const selectedMarker = markers?.find((m) => m.id === selectedMarkerId) ?? null
  const seek = (seconds: number) => playerRef.current?.seek(seconds)

  return (
    <div className="flex h-full">
      <div className="min-w-0 flex-1 space-y-4 overflow-y-auto p-6">
        <div>
          <Link to={`/games/${video.game_id}`} className="text-sm text-slate-400 hover:text-slate-600">
            ← Back to {game?.name || game?.location || 'game'}
          </Link>
          <h1 className="mt-1 text-xl font-semibold">{video.original_filename}</h1>
          <div className="mt-1 flex flex-wrap items-center gap-4">
            <p className="text-sm text-slate-500">Status: {video.status}</p>
            <GameSelector video={video} />
          </div>
        </div>

        {(pendingJobId || video.status === 'detecting') && (
          <JobStatusBanner jobId={pendingJobId} />
        )}
        {video.status === 'error' && video.detect_error && (
          <div className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-800">
            Detection failed: {video.detect_error}
          </div>
        )}

        {!video.calibration_profile_id && (
          <BasketCalibrationPrompt video={video} onJobQueued={setPendingJobId} />
        )}

        <VideoPlayer
          ref={playerRef}
          src={`/video/source/${video.gcs_relpath}`}
          onTimeUpdate={setCurrentTime}
          onDuration={setDuration}
        />

        <MarkerTimeline
          duration={duration}
          currentTime={currentTime}
          markers={markers ?? []}
          selectedMarkerId={selectedMarkerId}
          onSeek={seek}
          onSelectMarker={selectMarker}
        />

        <MarkerNav
          videoId={video.id}
          markers={markers ?? []}
          currentTime={currentTime}
          selectedMarkerId={selectedMarkerId}
          onSelectMarker={selectMarker}
          onSeek={seek}
        />
      </div>

      {selectedMarker && (
        <aside className="w-96 shrink-0 border-l border-slate-200 bg-white">
          <MarkerDetailPanel
            marker={selectedMarker}
            video={video}
            currentUser={currentUser}
            onClose={() => selectMarker(null)}
          />
        </aside>
      )}
    </div>
  )
}

// Each video's basket is usually in its own spot, so the default path is
// drawing the box right here on this video's own frame; reusing a saved one
// is the secondary option, for when the camera position genuinely repeated.
function BasketCalibrationPrompt({
  video,
  onJobQueued,
}: {
  video: Video
  onJobQueued: (jobId: string) => void
}) {
  const { data: calibrations } = useCalibrationProfiles()
  const createCalibration = useCreateCalibrationProfile()
  const attach = useAttachCalibrationProfile(video.id)
  const [box, setBox] = useState<{ bbox: [number, number, number, number]; w: number; h: number } | null>(null)

  const attachAndDetect = (calibrationId: number) =>
    attach.mutate(calibrationId, { onSuccess: (v) => v.job_id && onJobQueued(v.job_id) })

  const saveDrawn = () => {
    if (!box) return
    createCalibration.mutate(
      {
        // unique per video - names are a UNIQUE column, and this one is
        // tied to this specific video's camera position anyway
        name: `${video.original_filename} (#${video.id})`,
        hoop_bbox_norm: box.bbox,
        frame_width: box.w,
        frame_height: box.h,
      },
      { onSuccess: (calibration) => attachAndDetect(calibration.id) },
    )
  }

  const busy = createCalibration.isPending || attach.isPending

  return (
    <div className="space-y-3 rounded-md border border-amber-300 bg-amber-50 p-3">
      <p className="text-sm text-amber-900">
        Basket not calibrated yet. Drag a box around the basket below, then save to start auto-detection.
      </p>
      <HoopCalibrationCanvas
        imageSrc={calibrationFrameUrl(video.id)}
        onBoxChange={(bbox, w, h) => setBox({ bbox, w, h })}
      />
      <button
        type="button"
        disabled={!box || busy}
        onClick={saveDrawn}
        className="rounded-md bg-orange-500 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
      >
        {busy ? 'Saving…' : 'Save basket position & detect'}
      </button>
      {(createCalibration.isError || attach.isError) && (
        <p className="text-sm text-red-600">{((createCalibration.error || attach.error) as Error).message}</p>
      )}
      {calibrations && calibrations.length > 0 && (
        <div className="border-t border-amber-200 pt-3">
          <label className="mb-1 block text-xs text-amber-900">
            Or reuse a saved one (same camera position only):
          </label>
          <select
            defaultValue=""
            disabled={busy}
            onChange={(e) => {
              const id = Number(e.target.value)
              if (id) attachAndDetect(id)
            }}
            className="rounded-md border border-amber-400 bg-white px-2 py-1.5 text-sm"
          >
            <option value="">Select a saved basket calibration…</option>
            {calibrations.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  )
}

function GameSelector({ video }: { video: Video }) {
  const { data: games } = useGames()
  const reassign = useReassignGame(video.id)

  return (
    <label className="flex items-center gap-2 text-sm text-slate-500">
      Game:
      <select
        value={video.game_id}
        disabled={reassign.isPending}
        onChange={(e) => reassign.mutate(Number(e.target.value))}
        className="rounded-md border border-slate-300 bg-white px-2 py-1 text-sm text-slate-700"
      >
        {games?.map((g) => (
          <option key={g.id} value={g.id}>
            {g.name || `${g.location} · ${new Date(g.game_date).toLocaleDateString()}`}
          </option>
        ))}
      </select>
    </label>
  )
}
