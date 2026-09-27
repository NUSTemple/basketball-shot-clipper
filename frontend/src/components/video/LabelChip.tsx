import { useDeleteLabel } from '../../api/queries/labels'
import type { Label, LabelCategory } from '../../api/types'

interface LabelChipProps {
  label: Label
  category: LabelCategory | undefined
  canModerate: boolean
}

export function LabelChip({ label, category, canModerate }: LabelChipProps) {
  const deleteLabel = useDeleteLabel(label.marker_id)

  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-orange-100 px-2.5 py-1 text-xs font-medium text-orange-800">
      {category?.name ?? `category #${label.category_id}`}
      {label.player_name && <span className="text-orange-600">· {label.player_name}</span>}
      {canModerate && (
        <button
          type="button"
          aria-label="Remove label"
          disabled={deleteLabel.isPending}
          onClick={() => deleteLabel.mutate(label.id)}
          className="ml-0.5 text-orange-500 hover:text-orange-900"
        >
          ×
        </button>
      )}
    </span>
  )
}
