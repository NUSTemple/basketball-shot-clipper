import { useComments, useDeleteComment } from '../../api/queries/comments'
import type { Comment } from '../../api/types'

interface CommentListProps {
  markerId: number
  currentUserId: number | undefined
  canModerateAny: boolean
}

export function CommentList({ markerId, currentUserId, canModerateAny }: CommentListProps) {
  const { data: comments } = useComments(markerId)
  const deleteComment = useDeleteComment(markerId)

  if (!comments || comments.length === 0) {
    return <p className="text-sm text-slate-400">No comments yet.</p>
  }

  const canDelete = (comment: Comment) => canModerateAny || comment.user_id === currentUserId

  return (
    <ul className="space-y-2">
      {comments.map((comment) => (
        <li key={comment.id} className="rounded-md bg-slate-100 px-3 py-2 text-sm">
          <div className="flex items-start justify-between gap-2">
            <p className="whitespace-pre-wrap text-slate-800">{comment.text}</p>
            {canDelete(comment) && (
              <button
                type="button"
                aria-label="Delete comment"
                disabled={deleteComment.isPending}
                onClick={() => deleteComment.mutate(comment.id)}
                className="shrink-0 text-slate-400 hover:text-red-600"
              >
                ×
              </button>
            )}
          </div>
        </li>
      ))}
    </ul>
  )
}
