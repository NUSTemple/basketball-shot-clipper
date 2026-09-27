import { useState } from 'react'

import { useCreateComment } from '../../api/queries/comments'

export function CommentComposer({ markerId }: { markerId: number }) {
  const [text, setText] = useState('')
  const createComment = useCreateComment(markerId)

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = text.trim()
    if (!trimmed) return
    createComment.mutate(trimmed, { onSuccess: () => setText('') })
  }

  return (
    <form onSubmit={submit} className="flex gap-2">
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Add a comment…"
        className="flex-1 rounded-md border border-slate-300 px-2 py-1.5 text-sm"
      />
      <button
        type="submit"
        disabled={!text.trim() || createComment.isPending}
        className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
      >
        Post
      </button>
    </form>
  )
}
