import { Navigate, Route, Routes } from 'react-router-dom'

import { AppShell } from './components/layout/AppShell'
import { AdminCategoriesPage } from './pages/AdminCategoriesPage'
import { CalibrationProfilesPage } from './pages/CalibrationProfilesPage'
import { ExportPage } from './pages/ExportPage'
import { NotFoundPage } from './pages/NotFoundPage'
import { UploadPage } from './pages/UploadPage'
import { VideoDetailPage } from './pages/VideoDetailPage'
import { VideoLibraryPage } from './pages/VideoLibraryPage'

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/videos" replace />} />
        <Route path="videos" element={<VideoLibraryPage />} />
        <Route path="videos/:videoId" element={<VideoDetailPage />} />
        <Route path="upload" element={<UploadPage />} />
        <Route path="calibration-profiles" element={<CalibrationProfilesPage />} />
        <Route path="admin/categories" element={<AdminCategoriesPage />} />
        <Route path="export" element={<ExportPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  )
}
