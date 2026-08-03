"""device_config picks where inference runs and how big each batch is.

Both env overrides here were documented in docs/GPU_SETUP.md long before they
existed, so these tests exist partly to keep the docs honest. Everything runs
without torch installed and without a GPU of any kind: device_config imports
torch *inside* its functions, so a fake module in sys.modules is enough.
"""
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from shot_clipper import device_config as dc


def fake_torch(*, cuda=False, mps=False):
    t = MagicMock()
    t.cuda.is_available.return_value = cuda
    t.backends.mps.is_available.return_value = mps
    return t


def _clear(fn):
    """cache_clear if it's still the lru_cached original - a test may have
    swapped in a plain stub, which monkeypatch only restores after this runs."""
    getattr(fn, "cache_clear", lambda: None)()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(dc.DEVICE_ENV, raising=False)
    monkeypatch.delenv(dc.BATCH_SIZE_ENV, raising=False)
    _clear(dc._mac_chip_name)
    yield
    _clear(dc._mac_chip_name)


def _with_torch(monkeypatch, torch_mod):
    monkeypatch.setitem(sys.modules, "torch", torch_mod)


# ---------------------------------------------------------------- autodetect

@pytest.mark.parametrize("cuda,mps,expected", [
    (True, False, "cuda"),
    (True, True, "cuda"),    # cuda wins when both are somehow present
    (False, True, "mps"),
    (False, False, "cpu"),
])
def test_autodetect(monkeypatch, cuda, mps, expected):
    _with_torch(monkeypatch, fake_torch(cuda=cuda, mps=mps))
    assert dc.get_device() == expected


def test_missing_torch_is_cpu(monkeypatch):
    """The label UI installs without the ml extras, so torch may be absent -
    device selection has to survive that rather than blowing up at import."""
    import builtins
    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    assert dc.get_device() == "cpu"


def test_old_torch_without_mps_backend(monkeypatch):
    """torch < 1.12 has no backends.mps; the getattr guard must absorb that
    rather than raising past the ImportError handler."""
    t = MagicMock()
    t.cuda.is_available.return_value = False
    del t.backends.mps
    _with_torch(monkeypatch, t)
    assert dc.get_device() == "cpu"


# ------------------------------------------------------------ forced device

def test_forced_cpu_needs_no_torch_at_all(monkeypatch):
    monkeypatch.setenv(dc.DEVICE_ENV, "cpu")
    _with_torch(monkeypatch, fake_torch(cuda=True))
    assert dc.get_device() == "cpu"


def test_forced_device_that_is_available(monkeypatch):
    monkeypatch.setenv(dc.DEVICE_ENV, "mps")
    _with_torch(monkeypatch, fake_torch(mps=True))
    assert dc.get_device() == "mps"


def test_forced_device_that_is_unavailable_falls_back(monkeypatch, capsys):
    """Asking for cuda on a Mac must not crash deep inside torch later."""
    monkeypatch.setenv(dc.DEVICE_ENV, "cuda")
    _with_torch(monkeypatch, fake_torch(mps=True))
    assert dc.get_device() == "cpu"
    assert "falling back to CPU" in capsys.readouterr().out


def test_forced_nonsense_is_ignored(monkeypatch, capsys):
    monkeypatch.setenv(dc.DEVICE_ENV, "banana")
    _with_torch(monkeypatch, fake_torch(cuda=True))
    assert dc.get_device() == "cuda"
    assert "ignoring" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["CUDA", " cuda "])
def test_forced_device_is_case_and_space_tolerant(monkeypatch, value):
    monkeypatch.setenv(dc.DEVICE_ENV, value)
    _with_torch(monkeypatch, fake_torch(cuda=True))
    assert dc.get_device() == "cuda"


# --------------------------------------------------------------- batch size

@pytest.mark.parametrize("device,expected", [("cuda", 16), ("mps", 8), ("cpu", 4)])
def test_batch_size_defaults(device, expected):
    assert dc.get_optimal_batch_size(device) == expected


def test_batch_size_override(monkeypatch):
    monkeypatch.setenv(dc.BATCH_SIZE_ENV, "24")
    assert dc.get_optimal_batch_size("cuda") == 24


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "3.5"])
def test_batch_size_bad_override_falls_back(monkeypatch, capsys, bad):
    monkeypatch.setenv(dc.BATCH_SIZE_ENV, bad)
    assert dc.get_optimal_batch_size("mps") == 8
    assert "ignoring" in capsys.readouterr().out


# ------------------------------------------------------------------- warmup

def test_warmup_skips_cpu():
    model = MagicMock()
    dc.warmup_device(model, "cpu")
    model.predict.assert_not_called()


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_warmup_runs_and_synchronises(monkeypatch, device):
    t = fake_torch()
    _with_torch(monkeypatch, t)
    model = MagicMock()
    dc.warmup_device(model, device, imgsz=576)

    model.predict.assert_called_once()
    kwargs = model.predict.call_args.kwargs
    assert kwargs["device"] == device
    assert kwargs["imgsz"] == 576
    backend = t.cuda if device == "cuda" else t.mps
    backend.synchronize.assert_called_once()
    backend.empty_cache.assert_called_once()


def test_warmup_dummy_is_hwc_not_nchw(monkeypatch):
    """Ultralytics rejects a batched NCHW array outright. The original
    (1, 3, 640, 640) default made warmup a permanent silent no-op behind its
    own except clause."""
    _with_torch(monkeypatch, fake_torch())
    model = MagicMock()
    dc.warmup_device(model, "cuda", imgsz=320)
    dummy = model.predict.call_args.args[0]
    assert dummy.shape == (320, 320, 3)


def test_warmup_failure_is_not_fatal(monkeypatch, capsys):
    _with_torch(monkeypatch, fake_torch())
    model = MagicMock()
    model.predict.side_effect = ValueError("boom")
    dc.warmup_device(model, "cuda")           # must not raise
    assert "Warmup failed" in capsys.readouterr().out


# ------------------------------------------------------------------ summary

def test_summary_cpu():
    assert dc.device_summary("cpu") == "CPU"


def test_summary_cuda_names_the_card(monkeypatch):
    t = fake_torch(cuda=True)
    t.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4070"
    _with_torch(monkeypatch, t)
    assert dc.device_summary("cuda") == "CUDA (NVIDIA GeForce RTX 4070)"


def test_summary_mps_uses_the_chip_name(monkeypatch):
    monkeypatch.setattr(dc, "_mac_chip_name", lambda: "Apple M3 Max")
    assert dc.device_summary("mps") == "MPS (Apple M3 Max)"


def test_summary_mps_degrades_when_chip_unknown(monkeypatch):
    """Runs per job from label_ui/pipeline, so it must never raise or block."""
    monkeypatch.setattr(dc, "_mac_chip_name", lambda: "")
    assert dc.device_summary("mps") == "MPS (Apple Metal)"


def test_chip_name_is_empty_off_darwin(monkeypatch):
    monkeypatch.setattr(dc.sys, "platform", "win32")
    dc._mac_chip_name.cache_clear()
    assert dc._mac_chip_name() == ""


def test_chip_name_survives_a_broken_sysctl(monkeypatch):
    monkeypatch.setattr(dc.sys, "platform", "darwin")
    monkeypatch.setattr(dc.subprocess, "run",
                        MagicMock(side_effect=subprocess.TimeoutExpired("sysctl", 5)))
    dc._mac_chip_name.cache_clear()
    assert dc._mac_chip_name() == ""
