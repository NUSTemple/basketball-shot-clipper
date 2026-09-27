import { useState } from 'react'

import { useLabelCategories } from '../../api/queries/categories'
import { useCreateLabel } from '../../api/queries/labels'
import { useRoster } from '../../api/queries/roster'

export function LabelForm({ markerId }: { markerId: number }) {
  const { data: categories } = useLabelCategories()
  const { data: roster } = useRoster()
  const createLabel = useCreateLabel(markerId)

  const [categoryId, setCategoryId] = useState<number | ''>('')
  const [playerName, setPlayerName] = useState('')

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!categoryId) return
    createLabel.mutate(
      { category_id: categoryId, player_name: playerName.trim() || undefined },
      { onSuccess: () => setPlayerName('') },
    )
  }

  return (
    <form onSubmit={submit} className="flex flex-wrap items-center gap-2">
      <select
        value={categoryId}
        onChange={(e) => setCategoryId(e.target.value ? Number(e.target.value) : '')}
        className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
      >
        <option value="">Category…</option>
        {categories?.map((c) => (
          <option key={c.id} value={c.id}>
            {c.name}
          </option>
        ))}
      </select>
      <input
        list="roster-players"
        value={playerName}
        onChange={(e) => setPlayerName(e.target.value)}
        placeholder="Player (optional)"
        className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
      />
      <datalist id="roster-players">
        {roster?.map((name) => <option key={name} value={name} />)}
      </datalist>
      <button
        type="submit"
        disabled={!categoryId || createLabel.isPending}
        className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
      >
        Add label
      </button>
    </form>
  )
}
