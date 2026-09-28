import { NavLink } from 'react-router-dom'

import { useCurrentUser } from '../../api/queries/profile'

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `block rounded-md px-3 py-2 text-sm font-medium transition-colors ${
    isActive ? 'bg-orange-500 text-white' : 'text-slate-300 hover:bg-slate-800 hover:text-white'
  }`

export function NavSidebar() {
  const { data: user } = useCurrentUser()

  return (
    <nav className="flex flex-col gap-1 p-3">
      <NavLink to="/games" className={linkClass}>
        Games
      </NavLink>
      <NavLink to="/calibration-profiles" className={linkClass}>
        Calibration Profiles
      </NavLink>
      <NavLink to="/export" className={linkClass}>
        Export
      </NavLink>
      {user?.is_admin && (
        <NavLink to="/admin/categories" className={linkClass}>
          Admin: Categories
        </NavLink>
      )}
    </nav>
  )
}
