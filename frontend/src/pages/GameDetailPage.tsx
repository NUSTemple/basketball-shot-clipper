import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { useAddGamePlayer, useGame, useRemoveGamePlayer } from '../api/queries/games'
import { useRoster } from '../api/queries/roster'
import { type UploadStage, useUploadVideo } from '../api/queries/uploads'
import { useVideos } from '../api/queries/videos'
import type { VideoStatus } from '../api/types'
import { UploadDropzone } from '../components/upload/UploadDropzone'
import { formatTime } from '../lib/time'

const STATUS_STYLE: Record<VideoStatus, string> = {
  uploaded: 'bg-slate-100 text-slate-700',
  detecting: 'bg-blue-100 text-blue-800',
  ready: 'bg-green-100 text-green-800',
  error: 'bg-red-100 text-red-800',
}

const STAGE_LABEL: Record<UploadStage, string> = {
  'requesting-url': 'Requesting upload slot…',
  uploading: 'Uploading to storage…',
  confirming: 'Confirming upload…',
  registering: 'Registering video…',
}

export function GameDetailPage() {
  const { gameId: gameIdParam } = useParams<{ gameId: string }>()
  const gameId = Number(gameIdParam)

  const { data: game } = useGame(gameId)
  const { data: videos } = useVideos(gameId)

  if (!game) return <div className="p-6 text-slate-500">Loading…</div>

  const date = new Date(game.game_date).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })

  return (
    <div className="mx-auto max-w-4xl space-y-8 p-6">
      <div>
        <Link to="/games" className="text-sm text-slate-400 hover:text-slate-600">
          ← All games
        </Link>
        <h1 className="mt-1 text-xl font-semibold">{game.name || `${game.location} · ${date}`}</h1>
        <p className="text-sm text-slate-500">
          {game.location} · {date}
        </p>
        <Link to={`/export?gameId=${game.id}`} className="mt-2 inline-block text-sm text-orange-600 underline">
          Export this game's clips
        </Link>
      </div>

      <RosterSection gameId={gameId} players={game.players} />

      <div>
        <h2 className="mb-3 text-lg font-semibold">Upload a video</h2>
        <UploadSection gameId={gameId} />
      </div>

      <div>
        <h2 className="mb-3 text-lg font-semibold">Videos</h2>
        {videos?.length === 0 && <p className="text-sm text-slate-500">No videos in this game yet.</p>}
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
    </div>
  )
}

function RosterSection({ gameId, players }: { gameId: number; players: { id: number; name: string }[] }) {
  const { data: roster } = useRoster()
  const addPlayer = useAddGamePlayer(gameId)
  const removePlayer = useRemoveGamePlayer(gameId)
  const [name, setName] = useState('')

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = name.trim()
    if (!trimmed) return
    addPlayer.mutate(trimmed, { onSuccess: () => setName('') })
  }

  return (
    <section>
      <h2 className="mb-2 text-lg font-semibold">Players</h2>
      <div className="mb-2 flex flex-wrap gap-1.5">
        {players.map((p) => (
          <span
            key={p.id}
            className="inline-flex items-center gap-1.5 rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700"
          >
            {p.name}
            <button
              type="button"
              aria-label={`Remove ${p.name}`}
              onClick={() => removePlayer.mutate(p.id)}
              className="text-slate-400 hover:text-red-600"
            >
              ×
            </button>
          </span>
        ))}
        {players.length === 0 && <span className="text-sm text-slate-400">No players added yet.</span>}
      </div>
      <form onSubmit={submit} className="flex gap-2">
        <input
          list="roster-players"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Add a player…"
          className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
        />
        <datalist id="roster-players">
          {roster?.map((n) => <option key={n} value={n} />)}
        </datalist>
        <button
          type="submit"
          disabled={!name.trim() || addPlayer.isPending}
          className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
        >
          Add
        </button>
      </form>
    </section>
  )
}

function UploadSection({ gameId }: { gameId: number }) {
  const upload = useUploadVideo()
  const [stage, setStage] = useState<UploadStage | null>(null)

  const handleFile = (file: File) => {
    upload.mutate({ file, gameId, onStage: setStage })
  }

  return (
    <div>
      <UploadDropzone onFileSelected={handleFile} disabled={upload.isPending} />
      {stage && upload.isPending && <p className="mt-2 text-sm text-slate-600">{STAGE_LABEL[stage]}</p>}
      {upload.isError && <p className="mt-2 text-sm text-red-600">{(upload.error as Error).message}</p>}
      {upload.isSuccess && <p className="mt-2 text-sm text-green-700">Uploaded - see it below.</p>}
    </div>
  )
}
