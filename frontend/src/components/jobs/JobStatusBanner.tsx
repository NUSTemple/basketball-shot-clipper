import { useJob } from '../../api/queries/jobs'

const STYLES: Record<string, string> = {
  queued: 'bg-slate-100 text-slate-700 border-slate-300',
  running: 'bg-blue-50 text-blue-800 border-blue-300',
  done: 'bg-green-50 text-green-800 border-green-300',
  error: 'bg-red-50 text-red-800 border-red-300',
  cancelled: 'bg-slate-100 text-slate-600 border-slate-300',
}

export function JobStatusBanner({ jobId }: { jobId: string | null }) {
  const { data: job } = useJob(jobId)
  if (!jobId || !job) return null

  return (
    <div className={`rounded-md border px-3 py-2 text-sm ${STYLES[job.state] ?? STYLES.queued}`}>
      <span className="font-medium capitalize">{job.state}</span>
      {job.queue_position != null && job.queue_position > 0 && (
        <span> (position {job.queue_position} in queue)</span>
      )}
      {job.message && <span> — {job.message}</span>}
      {job.error && <span className="block font-mono text-xs">{job.error}</span>}
    </div>
  )
}
