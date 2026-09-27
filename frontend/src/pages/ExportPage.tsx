import { useState } from 'react'

import { useJob } from '../api/queries/jobs'
import { exportDownloadUrl, useCreateExport } from '../api/queries/exportJob'
import { isTerminalJobState } from '../api/types'
import { JobStatusBanner } from '../components/jobs/JobStatusBanner'
import { ExportFilterForm } from '../components/export/ExportFilterForm'

export function ExportPage() {
  const createExport = useCreateExport()
  const [jobId, setJobId] = useState<string | null>(null)
  const { data: job } = useJob(jobId)

  const noMatches = createExport.data && createExport.data.job_id == null

  return (
    <div className="mx-auto max-w-lg p-6">
      <h1 className="mb-4 text-xl font-semibold">Export Clips</h1>
      <p className="mb-4 text-sm text-slate-500">
        Filter markers by label and/or comment across every video, then cut and download matching
        clips as a zip.
      </p>

      <ExportFilterForm
        isPending={createExport.isPending}
        onSubmit={(filter) =>
          createExport.mutate(filter, {
            onSuccess: (result) => setJobId(result.job_id),
          })
        }
      />

      {noMatches && (
        <p className="mt-4 text-sm text-slate-500">No markers matched that filter.</p>
      )}

      {createExport.data && createExport.data.job_id && (
        <p className="mt-4 text-sm text-slate-600">
          Cutting {createExport.data.marker_count} marker(s) across {createExport.data.video_count}{' '}
          video(s)…
        </p>
      )}

      {jobId && (
        <div className="mt-4 space-y-3">
          <JobStatusBanner jobId={jobId} />
          {job && isTerminalJobState(job.state) && job.state === 'done' && (
            <a
              href={exportDownloadUrl(jobId)}
              className="inline-block rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white"
            >
              Download zip
            </a>
          )}
        </div>
      )}
    </div>
  )
}
