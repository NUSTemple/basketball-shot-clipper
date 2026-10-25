"""The vision-model step on machines without MLX: the Ollama backend, talked
to over its HTTP API, and how shot-clipper-highlights picks a backend."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from shot_clipper import highlight_vlm, highlights


@pytest.fixture
def fake_ollama(monkeypatch):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"models": [{"name": "qwen3-vl:30b"}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(req)
            self._send({"response": '<think>{"x": 1}</think>Sure: {"highlight_score": 7, "difficulty": 3}'})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(highlight_vlm, "OLLAMA_HOST", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(highlight_vlm, "OLLAMA_MODEL", "qwen3-vl:30b")
    yield seen
    server.shutdown()


def test_ollama_backend_sends_the_frames_and_parses_the_answer(fake_ollama, tmp_path, monkeypatch):
    assert highlight_vlm.ollama_has_model()
    monkeypatch.setattr(highlight_vlm, "grab_frame", lambda clip, t, out, vf: out.write_bytes(b"jpg"))
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps([{"clip": "V/shot_001.mp4", "path": "x.mp4", "hoop": [0.4, 0.3, 0.5, 0.4], "make": 5.0}]))
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr("sys.argv", ["highlight_vlm.py", str(jobs), str(out), "ollama"])
    highlight_vlm.main()
    row = json.loads(out.read_text())
    assert row["vlm"] == {"highlight_score": 7, "difficulty": 3}  # the <think> block isn't mistaken for it
    req = fake_ollama[0]
    assert req["model"] == "qwen3-vl:30b" and len(req["images"]) == 9
    assert "about t = 5.0s" in req["prompt"]
    highlight_vlm.main()  # resumable: nothing left to send
    assert len(fake_ollama) == 1


def test_no_ollama_model_means_no_backend(fake_ollama, monkeypatch):
    monkeypatch.setattr(highlight_vlm, "OLLAMA_MODEL", "qwen3-vl:8b")
    assert not highlight_vlm.ollama_has_model()


def test_backend_choice(monkeypatch, tmp_path):
    monkeypatch.delenv("SHOT_CLIPPER_VLM_BACKEND", raising=False)
    monkeypatch.setattr(highlight_vlm, "ollama_has_model", lambda: True)
    monkeypatch.setattr(highlights.sys, "platform", "win32")
    assert highlights.vlm_backend() == "ollama"
    monkeypatch.setattr(highlight_vlm, "ollama_has_model", lambda: False)
    assert highlights.vlm_backend() is None
    monkeypatch.setattr(highlights.sys, "platform", "darwin")
    monkeypatch.setattr(highlights, "VLM_PYTHON", tmp_path)  # exists
    assert highlights.vlm_backend() == "mlx"
    monkeypatch.setenv("SHOT_CLIPPER_VLM_BACKEND", "ollama")
    assert highlights.vlm_backend() == "ollama"
