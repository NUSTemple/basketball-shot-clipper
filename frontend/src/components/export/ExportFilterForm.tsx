import { useState } from 'react'

import { useLabelCategories } from '../../api/queries/categories'
import type { ExportFilter } from '../../api/queries/exportJob'

interface ExportFilterFormProps {
  onSubmit: (filter: ExportFilter) => void
  isPending: boolean
}

export function ExportFilterForm({ onSubmit, isPending }: ExportFilterFormProps) {
  const { data: categories } = useLabelCategories()
  const [selectedCategories, setSelectedCategories] = useState<Set<string>>(new Set())
  const [player, setPlayer] = useState('')
  const [commentKeyword, setCommentKeyword] = useState('')
  const [pre, setPre] = useState(5)
  const [post, setPost] = useState(2)

  const toggleCategory = (name: string) => {
    setSelectedCategories((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  const hasFilter = selectedCategories.size > 0 || player.trim() || commentKeyword.trim()

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!hasFilter) return
    onSubmit({
      categories: selectedCategories.size > 0 ? [...selectedCategories] : undefined,
      player: player.trim() || undefined,
      comment_keyword: commentKeyword.trim() || undefined,
      pre,
      post,
    })
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <div>
        <span className="mb-1 block text-sm font-medium text-slate-700">Label categories</span>
        <div className="flex flex-wrap gap-2">
          {categories?.map((c) => (
            <button
              key={c.id}
              type="button"
              onClick={() => toggleCategory(c.name)}
              className={`rounded-full border px-3 py-1 text-sm ${
                selectedCategories.has(c.name)
                  ? 'border-orange-500 bg-orange-500 text-white'
                  : 'border-slate-300 text-slate-700 hover:border-slate-400'
              }`}
            >
              {c.name}
            </button>
          ))}
        </div>
      </div>

      <div>
        <label className="mb-1 block text-sm font-medium text-slate-700">Player</label>
        <input
          value={player}
          onChange={(e) => setPlayer(e.target.value)}
          placeholder="Player name contains…"
          className="w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm"
        />
      </div>

      <div>
        <label className="mb-1 block text-sm font-medium text-slate-700">Comment keyword</label>
        <input
          value={commentKeyword}
          onChange={(e) => setCommentKeyword(e.target.value)}
          placeholder="Comment text contains…"
          className="w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm"
        />
      </div>

      <div className="flex gap-4">
        <label className="text-sm text-slate-700">
          Pre-roll (s)
          <input
            type="number"
            min={0}
            value={pre}
            onChange={(e) => setPre(Number(e.target.value))}
            className="ml-2 w-16 rounded-md border border-slate-300 px-2 py-1"
          />
        </label>
        <label className="text-sm text-slate-700">
          Post-roll (s)
          <input
            type="number"
            min={0}
            value={post}
            onChange={(e) => setPost(Number(e.target.value))}
            className="ml-2 w-16 rounded-md border border-slate-300 px-2 py-1"
          />
        </label>
      </div>

      <button
        type="submit"
        disabled={!hasFilter || isPending}
        className="rounded-md bg-orange-500 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
      >
        {isPending ? 'Starting export…' : 'Export clips'}
      </button>
      {!hasFilter && <p className="text-xs text-slate-400">Pick at least one filter.</p>}
    </form>
  )
}
