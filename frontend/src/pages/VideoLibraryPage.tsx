import { Link } from 'react-router-dom'

import { useVideos } from '../api/queries/videos'
import type { VideoStatus } from '../api/types'
import { formatTime } from '../lib/time'

const STATUS_STYLE: Record<VideoStatus, string> = {
  uploaded: 'bg-slate-100 text-slate-700',
  detecting: 'bg-blue-100 text-blue-800',
  ready: 'bg-green-100 text-green-800',
  error: 'bg-red-100 text-red-800',
}

export function VideoLibraryPage() {
  const { data: videos, isLoading } = useVideos()

  return (
    <div className="p-6">
      <h1 className="mb-4 text-xl font-semibold">Video Library</h1>
      {isLoading && <p className="text-slate-500">Loading…</p>}
      {videos?.length === 0 && (
        <p className="text-slate-500">
          No videos yet. <Link to="/upload" className="text-orange-600 underline">Upload one</Link> to get started.
        </p>
      )}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {videos?.map((video) => (
          <Link
            key={video.id}
            to={`/videos/${video.id}`}
            className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm transition-shadow hover:shadow-md"
          >
            <div className="mb-2 flex aspect-video items-center justify-center rounded-md bg-slate-100 text-3xl text-slate-300">
              🏀
            </div>
            <p className="truncate font-medium text-slate-900">{video.original_filename}</p>
            <div className="mt-1 flex items-center gap-2 text-xs text-slate-500">
              <span className={`rounded px-1.5 py-0.5 font-medium ${STATUS_STYLE[video.status]}`}>
                {video.status}
              </span>
              {video.duration_s != null && <span>{formatTime(video.duration_s)}</span>}
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}
