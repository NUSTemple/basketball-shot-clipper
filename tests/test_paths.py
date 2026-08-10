"""paths.py decides where every file the app touches lives.

Two things are worth pinning here. First, a checkout must keep resolving
./data and ./models exactly as it did when every path was a bare relative
Path - this refactor is not supposed to move anybody's existing labels or
calibrations. Second, a frozen build must put writable state somewhere the
installer does not replace on upgrade, and must never treat the unpacked
bundle (a temp directory PyInstaller deletes on exit) as writable.
"""
import sys

import pytest

from shot_clipper import paths


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (paths.HOME_ENV, paths.DATA_DIR_ENV, paths.MODELS_DIR_ENV,
                paths.CLIPS_DIR_ENV, paths.MEDIA_ROOT_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)


def _freeze(monkeypatch, bundle_dir):
    """Make paths believe it's running from a PyInstaller bundle."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_dir), raising=False)


# --------------------------------------------------------------------------
# checkout behaviour - the regression guard
# --------------------------------------------------------------------------

def test_checkout_data_dir_is_cwd_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert paths.data_dir() == tmp_path / "data"
    assert paths.configs_dir() == tmp_path / "data" / "configs"
    assert paths.jobs_dir() == tmp_path / "data" / "jobs"
    assert paths.dataset_dir() == tmp_path / "data" / "dataset"
    assert paths.models_dir() == tmp_path / "models"


def test_checkout_resource_dir_is_the_repo_root():
    """Not the package dir and not site-packages: bundled assets sit next to
    src/, so the walk up has to land on the repo root."""
    assert (paths.resource_dir() / "src" / "shot_clipper" / "paths.py").is_file()


def test_paths_follow_a_later_chdir(tmp_path, monkeypatch):
    """The old bare Path("data/configs") was resolved at syscall time, so it
    tracked the working directory. Functions preserve that; module-level
    constants computed at import would not."""
    monkeypatch.chdir(tmp_path)
    first = paths.configs_dir()
    sub = tmp_path / "elsewhere"
    sub.mkdir()
    monkeypatch.chdir(sub)
    assert paths.configs_dir() != first
    assert paths.configs_dir() == sub / "data" / "configs"


# --------------------------------------------------------------------------
# frozen behaviour
# --------------------------------------------------------------------------

def test_frozen_writable_state_leaves_the_bundle(tmp_path, monkeypatch):
    """The whole point: an upgrade replaces the install directory, and
    PyInstaller deletes _MEIPASS on exit. Either one would take the user's
    calibrations, labels and job history with it."""
    bundle = tmp_path / "bundle"
    _freeze(monkeypatch, bundle)
    monkeypatch.chdir(tmp_path)

    for directory in (paths.data_dir(), paths.models_dir()):
        assert not directory.is_relative_to(bundle)
        assert not directory.is_relative_to(tmp_path), "must not follow cwd either"
    assert paths.data_dir().is_relative_to(paths.user_data_root())


def test_frozen_resource_dir_is_the_unpacked_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    _freeze(monkeypatch, bundle)
    assert paths.resource_dir() == bundle


def test_frozen_without_meipass_falls_back_to_the_executable(tmp_path, monkeypatch):
    """A onedir build can be launched without _MEIPASS set; the files still
    sit next to the exe."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "shot-clipper.exe"))
    assert paths.resource_dir() == tmp_path


# --------------------------------------------------------------------------
# overrides
# --------------------------------------------------------------------------

def test_home_env_overrides_everything(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.HOME_ENV, str(tmp_path))
    assert paths.data_dir() == tmp_path / "data"
    assert paths.models_dir() == tmp_path / "models"


def test_data_dir_env_overrides_home(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.HOME_ENV, str(tmp_path / "home"))
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "elsewhere"))
    assert paths.data_dir() == tmp_path / "elsewhere"
    assert paths.models_dir() == tmp_path / "home" / "models"


def test_clips_and_media_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.CLIPS_DIR_ENV, str(tmp_path / "clips"))
    monkeypatch.setenv(paths.MEDIA_ROOT_ENV, str(tmp_path / "media"))
    assert paths.default_clips_dir() == tmp_path / "clips"
    assert paths.default_media_root() == tmp_path / "media"


def test_media_root_defaults_under_the_current_users_home():
    """It used to be hardcoded to one developer's macOS home directory, so
    the in-app folder browser opened on a path no other machine has."""
    assert paths.default_media_root().is_relative_to(paths.Path.home())
    assert paths.default_clips_dir().is_relative_to(paths.default_media_root())


# --------------------------------------------------------------------------
# model lookup
# --------------------------------------------------------------------------

def test_find_model_prefers_the_writable_copy(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    (bundle / "models").mkdir(parents=True)
    (bundle / "models" / "yolov8l.pt").write_bytes(b"bundled")
    _freeze(monkeypatch, bundle)
    monkeypatch.setenv(paths.MODELS_DIR_ENV, str(tmp_path / "user-models"))
    (tmp_path / "user-models").mkdir()

    assert paths.find_model("yolov8l.pt") == bundle / "models" / "yolov8l.pt"

    (tmp_path / "user-models" / "yolov8l.pt").write_bytes(b"newer")
    assert paths.find_model("yolov8l.pt") == tmp_path / "user-models" / "yolov8l.pt"


def test_find_model_missing_everywhere_points_at_the_writable_dir(tmp_path, monkeypatch):
    """Where ultralytics should download it to - never into the read-only
    bundle."""
    _freeze(monkeypatch, tmp_path / "empty-bundle")
    monkeypatch.setenv(paths.MODELS_DIR_ENV, str(tmp_path / "models"))
    assert paths.find_model("yolov8l.pt") == tmp_path / "models" / "yolov8l.pt"


def test_find_model_passes_an_explicit_path_straight_through(tmp_path, monkeypatch):
    """--model still means "this exact file", not "look this name up"."""
    monkeypatch.setenv(paths.MODELS_DIR_ENV, str(tmp_path / "models"))
    explicit = tmp_path / "somewhere" / "custom.pt"
    assert paths.find_model(str(explicit)) == explicit
    assert paths.find_model("models/yolov8l.pt") == paths.Path("models/yolov8l.pt")
