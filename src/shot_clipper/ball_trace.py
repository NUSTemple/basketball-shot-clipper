"""The ball's path around one shot, for drawing over the video.

detect_shots builds a full (t, x, y) ball track for the video and then
throws it away, keeping only the timestamps find_makes accepted. That
discards the evidence: when a make is missed there is no way to tell
whether YOLO never saw the ball or whether the geometry rejected a
perfectly good crossing. Same question a viewer asks looking at a
candidate - "what did it actually see?"

So this rebuilds the track for one shot's window, on demand, and caches it.

Rebuilt from the *original* video rather than the clip on disk: review
clips are 720p previews cut from the proxy (see clip_shots.cut_clip), and
ball recall falls off badly with the ball's pixel size - a trace drawn from
those would be sparse for reasons that have nothing to do with what
detection saw. The window is cut at delivery quality first, which is the
same footage run_detection scanned.

Coordinates are normalized to the frame (0-1), like everything else that
talks about hoop boxes here, so the same trace draws correctly over the
1080p original or the 720p proxy.
"""
import json
import tempfile
from pathlib import Path

from . import clip_shots, inference, paths
from .device_config import get_device

TRACES_DIRNAME = "_traces"


def traces_dir(clips_video_dir: Path) -> Path:
    return clips_video_dir / TRACES_DIRNAME


def trace_path(clips_video_dir: Path, clip_name: str) -> Path:
    return traces_dir(clips_video_dir) / f"{Path(clip_name).stem}.json"


def build(source_video: Path, timestamp: float, hoop_bbox_norm, *,
          pre: float = clip_shots.DEFAULT_PRE,
          post: float = clip_shots.DEFAULT_POST,
          detector=None, device: str | None = None, fps: float | None = None) -> dict:
    """Ball positions around `timestamp`, as {"points": [[t, x, y], ...]}.

    `t` is absolute video seconds, so the caller can line the trace up with
    a player's currentTime without knowing how the window was cut. Frames
    where no ball was found are omitted rather than stored as nulls - near
    the hoop that is most of them, and a gap in the trail is exactly what a
    reader should see.
    """
    from .features import build_ball_track_for_clip

    device = device or get_device()
    detector = detector or inference.load_detector(None, device=device)
    start = max(0.0, timestamp - pre)
    duration = (timestamp + post) - start

    with tempfile.TemporaryDirectory() as tmp:
        window = Path(tmp) / "window.mp4"
        clip_shots.cut_clip(source_video, start, duration, window, preview=False)
        kwargs = {"device": device}
        if fps:
            kwargs["fps"] = fps
        track = build_ball_track_for_clip(window, hoop_bbox_norm, detector, **kwargs)

    points = [[round(start + t, 3), round(x, 5), round(y, 5)]
              for t, x, y in track if x is not None]
    return {
        "t": round(float(timestamp), 2),
        "window": [round(start, 3), round(start + duration, 3)],
        "hoop_bbox_norm": [round(v, 5) for v in hoop_bbox_norm],
        "points": points,
        "n_frames": len(track),
        "n_detections": len(points),
    }


def load_or_build(clips_video_dir: Path, clip_name: str, source_video: Path,
                  timestamp: float, hoop_bbox_norm, **kwargs) -> dict:
    """Cached build(). The cache is keyed by clip name and lives beside the
    clips, so it is thrown away with them rather than outliving the cut it
    describes."""
    cached = trace_path(clips_video_dir, clip_name)
    if cached.is_file():
        try:
            return json.loads(cached.read_text())
        except (json.JSONDecodeError, OSError):
            pass  # unreadable cache is just a cache miss

    trace = build(source_video, timestamp, hoop_bbox_norm, **kwargs)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(trace))
    return trace


def hoop_for(source_video: Path):
    """The calibrated hoop box for a source video, or None if uncalibrated."""
    config_path = paths.find_config(source_video)
    if not config_path.is_file():
        return None
    return json.loads(config_path.read_text())["hoop_bbox_norm"]
