"""The hosted worker runs one job per /drain request and pings itself for the
next. A single request draining a long queue outlived Cloud Run's 60-minute
request timeout, and the instance was stopped under the job then running."""
import pytest

from shot_clipper.label_ui import jobs, worker_server
from shot_clipper.label_ui.jobstore import read_job, write_job


@pytest.fixture
def queue(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path)
    ran = []

    class Done:
        def __init__(self, job_file):
            self.job_file = job_file

        def wait(self):
            job = read_job(self.job_file)
            job["state"] = "done"
            write_job(self.job_file, job)

    def spawn(job_id, job_file):
        ran.append(job_id)
        return Done(job_file)

    monkeypatch.setattr(jobs, "_spawn_worker", spawn)
    for i in range(3):
        write_job(tmp_path / f"j{i}.json", {"id": f"j{i}", "state": "queued", "created_at": i})
    return ran


def test_capped_drain_runs_the_oldest_job_and_reports_more_queued(queue):
    assert jobs.drain_queue_once(max_jobs=1) is True
    assert queue == ["j0"]


def test_uncapped_drain_still_empties_the_queue(queue):
    assert jobs.drain_queue_once() is False
    assert queue == ["j0", "j1", "j2"]


def test_drain_endpoint_retriggers_only_while_work_remains(queue, monkeypatch):
    pings = []
    monkeypatch.setattr(worker_server, "_retrigger_later", lambda: pings.append(1))
    client = worker_server.app.test_client()

    for _ in range(3):
        assert client.post("/drain").get_json()["ok"]
    assert queue == ["j0", "j1", "j2"]
    assert len(pings) == 2  # after j0 and j1; nothing left after j2
    assert client.post("/drain").get_json()["more_queued"] is False
    assert len(pings) == 2
