import { useState } from 'react'
import { Link } from 'react-router-dom'

import { useCreateGame, useGames } from '../api/queries/games'

function formatGameLabel(game: { name: string | null; location: string; game_date: string }) {
  const date = new Date(game.game_date).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
  return game.name || `${game.location} · ${date}`
}

export function GamesListPage() {
  const { data: games, isLoading } = useGames()
  const createGame = useCreateGame()

  const [name, setName] = useState('')
  const [location, setLocation] = useState('')
  const [gameDate, setGameDate] = useState('')

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!location.trim() || !gameDate) return
    createGame.mutate(
      { location: location.trim(), game_date: gameDate, name: name.trim() || undefined },
      { onSuccess: () => { setName(''); setLocation(''); setGameDate('') } },
    )
  }

  return (
    <div className="p-6">
      <h1 className="mb-4 text-xl font-semibold">Games</h1>

      <form onSubmit={submit} className="mb-6 flex flex-wrap items-end gap-2 rounded-lg border border-slate-200 bg-white p-4">
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-600">Location</label>
          <input
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            placeholder="e.g. Home Court"
            className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
          />
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-600">Date</label>
          <input
            type="date"
            value={gameDate}
            onChange={(e) => setGameDate(e.target.value)}
            className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
          />
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-600">Name (optional)</label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Friday Pickup"
            className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
          />
        </div>
        <button
          type="submit"
          disabled={!location.trim() || !gameDate || createGame.isPending}
          className="rounded-md bg-orange-500 px-4 py-1.5 text-sm font-medium text-white disabled:opacity-50"
        >
          + New Game
        </button>
      </form>

      {isLoading && <p className="text-slate-500">Loading…</p>}
      {games?.length === 0 && <p className="text-slate-500">No games yet - create one above to get started.</p>}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {games?.map((game) => (
          <Link
            key={game.id}
            to={`/games/${game.id}`}
            className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm transition-shadow hover:shadow-md"
          >
            <p className="font-medium text-slate-900">{formatGameLabel(game)}</p>
            <p className="mt-1 text-sm text-slate-500">{game.location}</p>
            {game.players.length > 0 && (
              <p className="mt-2 truncate text-xs text-slate-400">
                {game.players.map((p) => p.name).join(', ')}
              </p>
            )}
          </Link>
        ))}
      </div>
    </div>
  )
}
