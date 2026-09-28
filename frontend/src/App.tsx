import { Navigate, Route, Routes } from 'react-router-dom'

import { AppShell } from './components/layout/AppShell'
import { AdminCategoriesPage } from './pages/AdminCategoriesPage'
import { CalibrationProfilesPage } from './pages/CalibrationProfilesPage'
import { ExportPage } from './pages/ExportPage'
import { GameDetailPage } from './pages/GameDetailPage'
import { GamesListPage } from './pages/GamesListPage'
import { NotFoundPage } from './pages/NotFoundPage'
import { VideoDetailPage } from './pages/VideoDetailPage'

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/games" replace />} />
        <Route path="games" element={<GamesListPage />} />
        <Route path="games/:gameId" element={<GameDetailPage />} />
        <Route path="videos/:videoId" element={<VideoDetailPage />} />
        <Route path="calibration-profiles" element={<CalibrationProfilesPage />} />
        <Route path="admin/categories" element={<AdminCategoriesPage />} />
        <Route path="export" element={<ExportPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  )
}
