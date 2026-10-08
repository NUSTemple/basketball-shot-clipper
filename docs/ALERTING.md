# Alerting (hosted deployment)

Problems with the GCP-hosted app are emailed to the operator by Cloud
Monitoring, instead of waiting for a user to report them. Everything lives in
project `basketball-shot-clipper` and carries the label
`app=basketball-shot-clipper`.

## Alert policies

All five are log-based, notify the "Shot Clipper alerts (email)" channel, and
are rate-limited to one email per 10 minutes per policy. They match every
Cloud Run service in the project (`label-ui`, `worker`, and the v2 pair).

| Policy | Fires on |
|---|---|
| container crashed or ran out of memory | Cloud Run's own "Out-of-memory event detected" / "Container terminated on signal" / "Memory limit of ... exceeded" lines |
| request failed with a server error (5xx) | any request log with `httpRequest.status>=500` |
| a video processing job failed | `jsonPayload.event="job_failed"` |
| job stuck in queue or worker could not be woken | `jsonPayload.event` is `job_stuck_queued` or `worker_trigger_failed` |
| unhandled exception (Python traceback) | `Traceback (most recent call last)` in a service's output |

## Where the app-level events come from

The first, second and last policies need nothing from the app. The other two
depend on `jobs.log_event()` in `src/shot_clipper/label_ui/jobs.py`, which on
Cloud Run writes one JSON line to stderr (Cloud Logging turns it into a
structured ERROR entry):

- `job_failed` - a job ended in `error`, or its worker died and the job was
  reaped. Carries `job_id`, `user`, `video`, `error`; the email shows the last
  three. The full traceback stays in the job's JSON under
  `gs://shot-clipper-data/_jobs/`.
- `job_stuck_queued` - a job has been queued for 15+ minutes while nothing is
  running. Detected when anyone's browser polls Job Status (or lists jobs), so
  a stuck job nobody is looking at is reported when the page is next opened,
  not before. label-ui re-sends the worker wake-up once when it reports this.
- `worker_trigger_failed` - label-ui's wake-up call to the worker errored or
  got an HTTP error back.

## Not covered

Failures that never reach the server: a browser upload dropping on a bad
connection, or a UI bug that produces no server error.

## Changing things

Policies and the channel were created through the Monitoring API and are
edited in Console under Monitoring > Alerting. To add a recipient, add a
notification channel and attach it to each policy.
