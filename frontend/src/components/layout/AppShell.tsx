import { Outlet } from 'react-router-dom'

import { NavSidebar } from './NavSidebar'
import { UserBadge } from './UserBadge'

export function AppShell() {
  return (
    <div className="flex h-full">
      <aside className="flex w-56 shrink-0 flex-col justify-between bg-slate-900">
        <div>
          <div className="px-4 py-4 text-lg font-semibold text-white">Shot Clipper</div>
          <NavSidebar />
        </div>
        <div className="p-3">
          <UserBadge />
        </div>
      </aside>
      <main className="min-w-0 flex-1 overflow-y-auto bg-slate-50">
        <Outlet />
      </main>
    </div>
  )
}
