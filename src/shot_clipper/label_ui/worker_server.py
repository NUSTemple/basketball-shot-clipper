"""HTTP entrypoint for the hosted worker Cloud Run service.

Unlike jobs.run_poller() (a permanently-running process for the local/Docker
two-container setup), this service is scaled to zero between jobs - a
request to /drain is what actually wakes it, and Cloud Run only bills for
as long as that request is in flight. label-ui's jobs.start_job() fires that
request (fire-and-forget, see jobs._trigger_worker) right after queuing a
job; this process then drains everything currently queued and lets the
response return, so the instance can scale back down.

Usage: shot-clipper-worker-server [--port 8080]
"""
import argparse
import threading

from flask import Flask, jsonify

from . import jobs

app = Flask(__name__)

# Belt-and-suspenders against running two GPU jobs at once: this service is
# deployed with --concurrency=1 --max-instances=1, which *should* already
# make that impossible, but jobs._trigger_worker() calls with a 1s read
# timeout (deliberately - it's fire-and-forget, not meant to wait for the
# whole drain) - confirmed live that a burst of job submissions (e.g. after
# batch-uploading several videos) can get multiple /drain requests routed
# to this same instance while an earlier one is still actually draining in
# its own thread (app.run(threaded=True)), each spawning its own worker
# subprocess onto the same GPU. Whether that's Cloud Run releasing the
# concurrency slot on client disconnect rather than on the handler
# returning, or something else, this lock makes correctness not depend on
# figuring out which: a concurrent call just returns immediately rather
# than draining twice - safe, since the drain loop that's already running
# re-globs the queue on every iteration and will pick up anything queued
# after this second call arrived anyway.
_drain_lock = threading.Lock()


@app.post("/drain")
def drain():
    if not _drain_lock.acquire(blocking=False):
        return jsonify({"ok": True, "already_draining": True})
    try:
        jobs.drain_queue_once()
    finally:
        _drain_lock.release()
    return jsonify({"ok": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
