import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { useCalibrationProfiles } from '../api/queries/calibrationProfiles'
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

interface UploadItem {
  id: string
  file: File
  stage: UploadStage
  loaded: number
  total: number
  etaSeconds: number | null
  error: string | null
  done: boolean
}

function UploadSection({ gameId }: { gameId: number }) {
  const upload = useUploadVideo()
  const { data: calibrations } = useCalibrationProfiles()
  const [calibrationId, setCalibrationId] = useState<number | ''>('')
  const [items, setItems] = useState<UploadItem[]>([])

  const update = (id: string, patch: Partial<UploadItem>) =>
    setItems((prev) => prev.map((it) => (it.id === id ? { ...it, ...patch } : it)))

  // Sequential, not parallel: several large video PUTs competing for the
  // same upstream bandwidth wouldn't finish sooner in parallel, and one at a
  // time keeps each file's percentage and time-left estimate meaningful.
  const processQueue = async (queue: UploadItem[], calibration: number | '') => {
    for (const item of queue) {
      try {
        await upload.mutateAsync({
          file: item.file,
          gameId,
          calibrationProfileId: calibration || undefined,
          onStage: (stage) => update(item.id, { stage }),
          onProgress: ({ loaded, total, etaSeconds }) => update(item.id, { loaded, total, etaSeconds }),
        })
        update(item.id, { done: true })
      } catch (err) {
        update(item.id, { error: (err as Error).message, done: true })
      }
    }
  }

  const handleFiles = (files: File[]) => {
    const newItems: UploadItem[] = files.map((file) => ({
      id: crypto.randomUUID(),
      file,
      stage: 'requesting-url',
      loaded: 0,
      total: file.size,
      etaSeconds: null,
      error: null,
      done: false,
    }))
    setItems((prev) => [...prev, ...newItems])
    void processQueue(newItems, calibrationId)
  }

  const anyPending = items.some((it) => !it.done)

  return (
    <div>
      <div className="mb-3">
        <label className="mb-1 block text-xs font-medium text-slate-600">Basket calibration</label>
        <select
          value={calibrationId}
          onChange={(e) => setCalibrationId(e.target.value ? Number(e.target.value) : '')}
          disabled={anyPending}
          className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
        >
          <option value="">Skip - calibrate each video after upload</option>
          {calibrations?.map((c) => (
            <option key={c.id} value={c.id}>
              Reuse: {c.name}
            </option>
          ))}
        </select>
        <p className="mt-1 text-xs text-slate-400">
          Only reuse one if these videos were shot from the same camera position.
        </p>
      </div>
      <UploadDropzone onFilesSelected={handleFiles} disabled={anyPending} />
      {items.length > 0 && (
        <ul className="mt-3 space-y-2">
          {items.map((item) => (
            <UploadItemRow key={item.id} item={item} />
          ))}
        </ul>
      )}
    </div>
  )
}

function UploadItemRow({ item }: { item: UploadItem }) {
  const pct = item.total > 0 ? Math.min(100, Math.round((item.loaded / item.total) * 100)) : 0
  const uploading = item.stage === 'uploading' && !item.done

  return (
    <li className="rounded-md border border-slate-200 bg-white px-3 py-2 text-sm">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate font-medium text-slate-800">{item.file.name}</span>
        {!item.error && (
          <span className="shrink-0 text-xs text-slate-500">
            {item.done ? 'Done' : uploading ? `${pct}%` : STAGE_LABEL[item.stage]}
          </span>
        )}
      </div>
      {uploading && (
        <>
          <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
            <div className="h-full rounded-full bg-orange-500 transition-all" style={{ width: `${pct}%` }} />
          </div>
          <p className="mt-1 text-xs text-slate-400">
            {formatBytes(item.loaded)} of {formatBytes(item.total)}
            {item.etaSeconds != null && ` · about ${formatTime(item.etaSeconds)} left`}
          </p>
        </>
      )}
      {item.error && <p className="mt-1 text-xs text-red-600">{item.error}</p>}
    </li>
  )
}

function formatBytes(bytes: number) {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  return `${Math.round(bytes / 1024)} KB`
}
