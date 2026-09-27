import { useCurrentUser } from '../../api/queries/profile'

export function UserBadge() {
  const { data: user } = useCurrentUser()
  if (!user) return null

  return (
    <div className="flex items-center gap-2 rounded-md bg-slate-800 px-3 py-2 text-sm text-slate-200">
      <span className="flex size-6 items-center justify-center rounded-full bg-orange-500 text-xs font-semibold text-white">
        {(user.display_name || user.email)[0]?.toUpperCase()}
      </span>
      <span className="truncate">{user.display_name || user.email}</span>
      {user.is_admin && (
        <span className="rounded bg-slate-700 px-1.5 py-0.5 text-xs text-slate-300">admin</span>
      )}
    </div>
  )
}
