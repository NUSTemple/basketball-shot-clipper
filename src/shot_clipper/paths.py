"""Where shot-clipper reads and writes files.

Every path used to be a bare relative Path - Path("data/configs"),
"models/yolov8l.pt" - resolved against the process's current directory. That
works exactly as long as everything runs from the repo root, which is why
installer/launch-label-ui.bat has to `cd /d "%~dp0.."` before starting the
app, and why running a tool from anywhere else silently created a second,
empty data root.

A frozen (PyInstaller) build breaks that assumption outright, so paths come
from here instead, split into two roots that are the same directory today
and different directories once frozen:

- resource_dir() - read-only files shipped *with* the app: bundled model
  weights, bundled ffmpeg. Under PyInstaller this is the unpacked bundle
  (sys._MEIPASS), which is a temp directory that is deleted on exit and must
  never be written to. In a checkout it's the repo root.
- app_root() - everything the user creates: calibrations, labels, job
  records, trained filters. Frozen, this is a per-user directory outside the
  install (%LOCALAPPDATA%\\shot-clipper on Windows), because an upgrade
  replaces the install directory wholesale and would take the user's
  calibrations and labels with it.

Unfrozen, app_root() is the current working directory, so a checkout keeps
reading and writing ./data and ./models exactly as before - no migration,
and Docker's WORKDIR /app keeps resolving to /app/data.

Every path is a function rather than a module-level constant on purpose:
constants would freeze the answer at import time, before main() has had a
chance to read $SHOT_CLIPPER_HOME, and would make a test's tmp_path
unreachable. dataset_labels.labels_path() and roster.roster_path() already
worked this way.
"""
import os
import sys
from pathlib import Path

HOME_ENV = "SHOT_CLIPPER_HOME"
DATA_DIR_ENV = "SHOT_CLIPPER_DATA_DIR"
MODELS_DIR_ENV = "SHOT_CLIPPER_MODELS_DIR"
CLIPS_DIR_ENV = "SHOT_CLIPPER_CLIPS_DIR"
MEDIA_ROOT_ENV = "SHOT_CLIPPER_MEDIA_ROOT"

# Which YOLO weights detection and feature extraction default to. yolov8l,
# not yolov8m: on a distant camera yolov8m saw the ball in only 4.7% of
# sampled frames vs 9.1%, and the trajectory rule needs two sightings to
# fire at all (see detect_shots.main's --model help).
DEFAULT_DETECT_WEIGHTS = "yolov8l.pt"


def is_frozen() -> bool:
    """Are we running from a PyInstaller-style bundle rather than a checkout?"""
    return bool(getattr(sys, "frozen", False))


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def resource_dir() -> Path:
    """Root of the read-only files shipped with the app.

    Frozen, PyInstaller unpacks the bundle to sys._MEIPASS and points
    sys.executable at the launcher instead, so neither one alone covers both
    onedir and onefile layouts - _MEIPASS is the correct answer when it
    exists. This directory is deleted when the process exits; nothing may be
    written here.
    """
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        return Path(meipass) if meipass else Path(sys.executable).resolve().parent
    # src/shot_clipper/paths.py -> src/shot_clipper -> src -> repo root
    return Path(__file__).resolve().parents[2]


def user_data_root() -> Path:
    """Per-user, per-platform directory for a frozen install's writable state.

    Only consulted when frozen: a checkout keeps using the working directory
    (see app_root) so existing repos and the Docker image are unaffected.
    """
    if sys.platform == "win32":
        base = _env_path("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = _env_path("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return base / "shot-clipper"


def app_root() -> Path:
    """Root of everything the user creates. $SHOT_CLIPPER_HOME overrides it."""
    return _env_path(HOME_ENV) or (user_data_root() if is_frozen() else Path.cwd())


def data_dir() -> Path:
    return _env_path(DATA_DIR_ENV) or app_root() / "data"


def configs_dir() -> Path:
    """Hoop calibrations, one <video_stem>.json per source video."""
    return data_dir() / "configs"


def ground_truth_dir() -> Path:
    """Detection output, one <video_stem>_detected.json per source video."""
    return data_dir() / "ground_truth"


def dataset_dir() -> Path:
    """labels.json, roster.json, and the materialized dataset folders."""
    return data_dir() / "dataset"


def jobs_dir() -> Path:
    return data_dir() / "jobs"


def thumbnail_cache_dir() -> Path:
    return data_dir() / "thumbnails_cache"


def config_path_for(video_path: Path) -> Path:
    return configs_dir() / f"{Path(video_path).stem}.json"


def ground_truth_path_for(video_path: Path) -> Path:
    return ground_truth_dir() / f"{Path(video_path).stem}_detected.json"


def models_dir() -> Path:
    """Writable models directory: the trained shot filter, plus any YOLO
    weights ultralytics downloads at runtime. Deliberately not
    resource_dir()/models - that one is read-only once frozen."""
    return _env_path(MODELS_DIR_ENV) or app_root() / "models"


def bundled_models_dir() -> Path:
    return resource_dir() / "models"


def find_model(name: str) -> Path:
    """Locate a model file by bare filename, preferring the user's copy.

    A frozen build ships weights inside the bundle, but a user can also have
    a newer or hand-placed copy (and train_filter writes shot_filter.joblib
    into models_dir() at runtime), so the writable location wins. When
    neither exists the writable path is returned anyway: that is where
    ultralytics should download to, and where callers should report a
    missing file from.
    """
    if Path(name).is_absolute() or len(Path(name).parts) > 1:
        return Path(name).expanduser()  # an explicit path is used as given
    writable = models_dir() / name
    if writable.is_file():
        return writable
    bundled = bundled_models_dir() / name
    if bundled.is_file():
        return bundled
    return writable


def detect_weights() -> Path:
    return find_model(DEFAULT_DETECT_WEIGHTS)


def filter_model_path() -> Path:
    return find_model("shot_filter.joblib")


def filter_meta_path() -> Path:
    return find_model("shot_filter_meta.json")


def default_media_root() -> Path:
    """Where the in-app folder browser starts and stays confined to.

    Source videos and clip output both normally live under the user's videos
    folder, so the picker has no reason to wander into unrelated system
    directories. This used to be hardcoded to one developer's home directory
    on macOS, which meant the browser opened on a path that does not exist on
    any other machine.
    """
    env = _env_path(MEDIA_ROOT_ENV)
    if env:
        return env
    home = Path.home()
    if sys.platform == "darwin":
        movies = home / "Movies"
        if movies.is_dir():
            return movies
    return home / "Videos"


def default_clips_dir() -> Path:
    """Where cut clips land unless the user picks somewhere else. Matches the
    folder installer/launch-label-ui.bat creates and passes in."""
    return _env_path(CLIPS_DIR_ENV) or default_media_root() / "shot-clipper" / "clips"
