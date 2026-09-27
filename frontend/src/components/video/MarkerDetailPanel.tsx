import { useLabelCategories } from '../../api/queries/categories'
import { useLabels } from '../../api/queries/labels'
import { useDeleteMarker, useUpdateMarkerState } from '../../api/queries/markers'
import type { Marker, Profile, Video } from '../../api/types'
import { formatTime } from '../../lib/time'
import { CommentComposer } from './CommentComposer'
import { CommentList } from './CommentList'
import { LabelChip } from './LabelChip'
import { LabelForm } from './LabelForm'

interface MarkerDetailPanelProps {
  marker: Marker
  video: Video
  currentUser: Profile | undefined
  onClose: () => void
}

export function MarkerDetailPanel({ marker, video, currentUser, onClose }: MarkerDetailPanelProps) {
  const { data: labels } = useLabels(marker.id)
  const { data: categories } = useLabelCategories()
  const updateState = useUpdateMarkerState(video.id)
  const deleteMarker = useDeleteMarker(video.id)

  const isModerator = !!currentUser && (currentUser.is_admin || currentUser.id === video.owner_user_id)
  const canDeleteMarker = (labels?.length ?? 0) === 0

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-4">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-lg font-semibold">{formatTime(marker.timestamp_s)}</h2>
          <p className="text-xs text-slate-500">
            {marker.source === 'auto' ? 'Auto-detected' : 'Manually added'} · {marker.state}
          </p>
        </div>
        <button type="button" onClick={onClose} className="text-slate-400 hover:text-slate-700">
          Close
        </button>
      </div>

      {marker.source === 'auto' && marker.state === 'unconfirmed' && (
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => updateState.mutate({ markerId: marker.id, state: 'confirmed' })}
            className="flex-1 rounded-md bg-green-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-green-700"
          >
            Confirm
          </button>
          <button
            type="button"
            onClick={() => updateState.mutate({ markerId: marker.id, state: 'dismissed' })}
            className="flex-1 rounded-md bg-slate-200 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-300"
          >
            Dismiss
          </button>
        </div>
      )}

      <section>
        <h3 className="mb-2 text-sm font-semibold text-slate-700">Labels</h3>
        <div className="mb-2 flex flex-wrap gap-1.5">
          {labels?.map((label) => (
            <LabelChip
              key={label.id}
              label={label}
              category={categories?.find((c) => c.id === label.category_id)}
              canModerate={isModerator || label.user_id === currentUser?.id}
            />
          ))}
          {labels?.length === 0 && <span className="text-sm text-slate-400">No labels yet.</span>}
        </div>
        <LabelForm markerId={marker.id} />
      </section>

      <section className="flex-1">
        <h3 className="mb-2 text-sm font-semibold text-slate-700">Comments</h3>
        <div className="mb-2">
          <CommentList markerId={marker.id} currentUserId={currentUser?.id} canModerateAny={isModerator} />
        </div>
        <CommentComposer markerId={marker.id} />
      </section>

      {canDeleteMarker && (
        <button
          type="button"
          onClick={() => deleteMarker.mutate(marker.id, { onSuccess: onClose })}
          className="text-left text-xs text-red-500 hover:text-red-700"
        >
          Delete this marker
        </button>
      )}
    </div>
  )
}
