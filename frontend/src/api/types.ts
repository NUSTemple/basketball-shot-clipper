// Mirrors each api/*.py file's _serialize() exactly - see the Python route
// file named in each comment for the source of truth.

// api/profile.py
export interface Profile {
  id: number
  email: string
  display_name: string | null
  avatar_url: string | null
  is_admin: boolean
}

// api/calibration_profiles.py
export interface CalibrationProfile {
  id: number
  name: string
  hoop_bbox_norm: [number, number, number, number]
  frame_width: number | null
  frame_height: number | null
  created_at: string | null
}

// api/videos.py
export type VideoStatus = 'uploaded' | 'detecting' | 'ready' | 'error'

export interface Video {
  id: number
  owner_user_id: number
  gcs_relpath: string
  original_filename: string
  calibration_profile_id: number | null
  duration_s: number | null
  status: VideoStatus
  detect_error: string | null
  created_at: string | null
  detected_at: string | null
  // POST/PATCH only - the detect_markers job just queued, if any
  job_id?: string | null
}

// api/markers.py
export type MarkerSource = 'auto' | 'manual'
export type MarkerState = 'unconfirmed' | 'confirmed' | 'dismissed'

export interface Marker {
  id: number
  video_id: number
  timestamp_s: number
  source: MarkerSource
  state: MarkerState
  created_by_user_id: number | null
  confirmed_by_user_id: number | null
  confirmed_at: string | null
  created_at: string | null
}

// api/labels.py - one row per (marker, user, category, player); never
// overwritten by another user's row
export interface Label {
  id: number
  marker_id: number
  user_id: number
  category_id: number
  player_name: string | null
  created_at: string | null
}

// api/comments.py - flat, no threading
export interface Comment {
  id: number
  marker_id: number
  user_id: number
  text: string
  created_at: string | null
  updated_at: string | null
}

// api/categories.py - admin-managed
export interface LabelCategory {
  id: number
  name: string
  active: boolean
}

// api/export.py
export interface ExportResult {
  job_id: string | null
  video_count: number
  marker_count: number
}

// api/jobs.py - a thin wrapper over jobs.get_job(), whose dict shape is a
// grab-bag the worker fills in as it goes (see label_ui/jobs.py/worker.py) -
// typed for the fields every screen actually reads, with an index escape
// hatch for the rest rather than pretending to fully model it.
export type JobState = 'queued' | 'running' | 'done' | 'error' | 'cancelled'

export interface Job {
  id: string
  kind: string
  state: JobState
  message: string | null
  error: string | null
  queue_position: number | null
  cut_clip_ids?: number[]
  n_markers?: number
  duration_s?: number | null
  [key: string]: unknown
}

export function isTerminalJobState(state: JobState): boolean {
  return state === 'done' || state === 'error' || state === 'cancelled'
}
