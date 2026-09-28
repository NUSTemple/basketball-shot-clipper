import { useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { useCalibrationProfiles } from '../api/queries/calibrationProfiles'
import { useGame } from '../api/queries/games'
import { useCurrentUser } from '../api/queries/profile'
import { useAttachCalibrationProfile, useVideo } from '../api/queries/videos'
import { useMarkers } from '../api/queries/markers'
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
          <p className="text-sm text-slate-500">Status: {video.status}</p>
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
          <CalibrationProfilePrompt videoId={video.id} onJobQueued={setPendingJobId} />
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

function CalibrationProfilePrompt({
  videoId,
  onJobQueued,
}: {
  videoId: number
  onJobQueued: (jobId: string) => void
}) {
  const { data: profiles } = useCalibrationProfiles()
  const attach = useAttachCalibrationProfile(videoId)

  return (
    <div className="rounded-md border border-amber-300 bg-amber-50 p-3">
      <p className="mb-2 text-sm text-amber-900">
        No calibration profile attached yet - pick one to start background auto-detection.
      </p>
      <select
        defaultValue=""
        onChange={(e) => {
          const id = Number(e.target.value)
          if (!id) return
          attach.mutate(id, { onSuccess: (video) => video.job_id && onJobQueued(video.job_id) })
        }}
        className="rounded-md border border-amber-400 bg-white px-2 py-1.5 text-sm"
      >
        <option value="">Select a calibration profile…</option>
        {profiles?.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name}
          </option>
        ))}
      </select>
    </div>
  )
}
