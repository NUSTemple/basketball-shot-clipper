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

from flask import Flask, jsonify

from . import jobs

app = Flask(__name__)


@app.post("/drain")
def drain():
    jobs.drain_queue_once()
    return jsonify({"ok": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
