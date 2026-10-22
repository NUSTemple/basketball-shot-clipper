"""Player identification: grouping/naming logic, the scan's goal selection and
gallery, and the label UI's Players API."""
import json

import numpy as np
import pytest

from shot_clipper.dataset_labels import load_labels
from shot_clipper.label_ui import app as app_module
from shot_clipper.person_id import grouping, scan
from shot_clipper.person_id.shooter import people, pick_shooter


def _look(seed, n=1, noise=0.02):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=64)
    return [grouping.unit(base + rng.normal(scale=noise, size=64) * np.linalg.norm(base)) for _ in range(n)]


# ---------------------------------------------------------------- grouping

def test_groups_follow_appearance_largest_first():
    embs = _look(1, 5) + _look(2, 2)
    assert grouping.group(embs) == [0] * 5 + [1] * 2
    assert grouping.group([]) == [] and grouping.group(_look(3)) == [0]


def test_a_tagged_member_names_its_whole_group():
    a = _look(1, 3)
    named = grouping.name_groups({0: [(a[0], None), (a[1], "Allen"), (a[2], None)]}, gallery=[])
    assert named[0]["name"] == "Allen" and named[0]["how"] == "tagged"


def test_gallery_names_a_group_only_with_a_clear_margin():
    allen, eddie = _look(1, 4), _look(2, 4)
    gallery = ([{"session": "0926", "name": "Allen", "emb": e} for e in allen[:3]]
               + [{"session": "0926", "name": "Eddie", "emb": e} for e in eddie[:3]])
    named = grouping.name_groups({0: [(allen[3], None)], 1: [(eddie[3], None)]}, gallery)
    assert (named[0]["name"], named[1]["name"]) == ("Allen", "Eddie")
    assert named[0]["how"] == "matched"
    stranger = _look(3)[0]  # nobody the gallery knows
    assert grouping.name_groups({0: [(stranger, None)]}, gallery[:3])[0]["name"] is None
    halfway = grouping.unit(grouping.unit(np.mean(allen[:3], axis=0)) + grouping.unit(np.mean(eddie[:3], axis=0)))
    assert grouping.name_groups({0: [(halfway, None)]}, gallery)[0]["name"] is None


def test_a_name_keeps_one_look_per_week():
    week1, week2 = _look(1, 2), _look(2, 2)
    protos = grouping.prototypes([{"session": "a", "name": "Allen", "emb": e} for e in week1]
                                 + [{"session": "b", "name": "Allen", "emb": e} for e in week2])
    assert len(protos["Allen"]) == 2


# ---------------------------------------------------------------- shooter

def test_merged_person_box_keeps_the_scorer_label():
    dets = [("player", 0.9, [0, 0, 10, 30]), ("scorer", 0.6, [0, 0, 10, 31]), ("ball", 0.8, [4, 2, 6, 4])]
    assert people(dets) == [("scorer", 0.6, [0, 0, 10, 30])]


def test_shooter_prefers_the_scorer_with_the_ball():
    times = [3.0, 3.5]
    dets = [[("scorer", 0.9, [100, 0, 110, 30])],
            [("scorer", 0.6, [0, 0, 10, 30]), ("ball", 0.8, [4, 2, 6, 4])]]
    fi, box, how = pick_shooter(times, dets, make=5.0)
    assert (fi, box, how) == (1, [0, 0, 10, 30], "scorer+ball")
    assert pick_shooter(times, [[], []], make=5.0) is None


# ---------------------------------------------------------------- scan helpers

def test_goal_selection_trusts_labels_then_ai_scores(tmp_path):
    v = tmp_path / "DJI_20261017_0001"
    v.mkdir()
    for i in range(1, 5):
        (v / f"shot_00{i}.mp4").write_bytes(b"x")
    (v / "_prelabel_scores.json").write_text(json.dumps({"shot_001.mp4": 0.9, "shot_002.mp4": 0.9,
                                                           "shot_003.mp4": 0.1, "shot_004.mp4": 0.1}))
    labels = {"DJI_20261017_0001/shot_002.mp4": {"label": "no_goal"},
              "DJI_20261017_0001/shot_003.mp4": {"label": "goal"}}
    key = lambda p: str(p.relative_to(tmp_path))  # noqa: E731
    assert [c.name for c in scan.goal_clips(v, labels, key)] == ["shot_001.mp4", "shot_003.mp4"]
    (v / "_prelabel_scores.json").unlink()
    assert len(scan.goal_clips(v, {}, key)) == 4  # nothing known: try every clip
    assert scan.session_of("acct/DJI_20261017170711_0001_D") == "20261017"


def test_scan_groups_names_and_feeds_the_gallery_without_rerunning_the_models(tmp_path):
    root = tmp_path / "clips"
    v = root / "DJI_20261017_0001"
    v.mkdir(parents=True)
    looks = _look(1, 3) + _look(2, 2)
    clips = {}
    for i, e in enumerate(looks, start=1):
        (v / f"shot_00{i}.mp4").write_bytes(b"x")
        clips[f"shot_00{i}.mp4"] = {"emb": [float(x) for x in e], "t": 3.0, "box": [0, 0, 1, 1], "how": "scorer"}
    from shot_clipper.person_id import reid
    (v / "_players.json").write_text(json.dumps({"method": reid.METHOD, "clips": clips}))
    labels = {f"DJI_20261017_0001/shot_00{i}.mp4": {"label": "goal"} for i in range(1, 6)}
    labels["DJI_20261017_0001/shot_001.mp4"]["scorer"] = "Allen"
    gallery = tmp_path / "gallery.jsonl"
    summary = scan.scan(root, labels, detector=object(), model=object(), gallery_path=gallery, log=lambda m: None)
    assert summary["20261017"] == {"goals": 5, "groups": 2, "named": 3, "tagged": 3}
    out = json.loads((v / "_players.json").read_text())["clips"]
    assert {out[f"shot_00{i}.mp4"]["suggested"] for i in (1, 2, 3)} == {"Allen"}
    assert out["shot_004.mp4"]["group"] == out["shot_005.mp4"]["group"] != out["shot_001.mp4"]["group"]
    assert [json.loads(line)["name"] for line in gallery.read_text().splitlines()] == ["Allen"]


# ---------------------------------------------------------------- label UI API

@pytest.fixture
def env(tmp_path, monkeypatch):
    clips_dir = tmp_path / "clips"
    v = clips_dir / "DJI_20261017_0001"
    v.mkdir(parents=True)
    for i in range(1, 5):
        (v / f"shot_00{i}.mp4").write_bytes(b"x")
    (v / "_players.json").write_text(json.dumps({"method": "x", "clips": {
        "shot_001.mp4": {"group": "20261017-00", "suggested": "Allen", "suggested_how": "matched", "confidence": 0.05},
        "shot_002.mp4": {"group": "20261017-00", "suggested": "Allen", "suggested_how": "matched", "confidence": 0.05},
        "shot_003.mp4": {"group": "20261017-01", "suggested": None, "suggested_how": None, "confidence": 0.0},
        "shot_009.mp4": {"group": "20261017-01"},  # clip since deleted: not listed
    }}))
    base = tmp_path / "dataset"
    base.mkdir()
    (base / "labels.json").write_text(json.dumps({"DJI_20261017_0001/shot_002.mp4": {"label": "goal", "stars": 4},
                                                  "DJI_20261017_0001/shot_003.mp4": {"label": "no_goal"}}))
    app_module.app.config["CLIPS_DIR"] = clips_dir
    monkeypatch.setattr(app_module, "dataset_base_for_request", lambda: base)
    with app_module.app.test_client() as c:
        yield c, base


def test_players_lists_groups_per_session_with_current_tags(env):
    client, _ = env
    sessions = client.get("/api/players").get_json()["sessions"]
    assert [s["session"] for s in sessions] == ["20261017"]
    g0, g1 = sessions[0]["groups"]
    assert g0["suggested"] == "Allen" and [c["path"] for c in g0["clips"]] == [
        "DJI_20261017_0001/shot_001.mp4", "DJI_20261017_0001/shot_002.mp4"]
    assert g0["clips"][1]["stars"] == 4
    assert [c["path"] for c in g1["clips"]] == ["DJI_20261017_0001/shot_003.mp4"]


def test_naming_a_group_tags_goals_once_and_never_overrides_a_miss(env):
    client, base = env
    resp = client.post("/api/players/name", json={"name": " Allen ", "clips": [
        "DJI_20261017_0001/shot_001.mp4", "DJI_20261017_0001/shot_002.mp4", "DJI_20261017_0001/shot_003.mp4"]})
    assert resp.get_json() == {"ok": True, "named": 2, "skipped": 1}
    labels = load_labels(base)
    assert labels["DJI_20261017_0001/shot_001.mp4"] == {"label": "goal", "source": "ai", "scorer": "Allen"}
    assert labels["DJI_20261017_0001/shot_002.mp4"] == {"label": "goal", "stars": 4, "scorer": "Allen"}
    assert labels["DJI_20261017_0001/shot_003.mp4"] == {"label": "no_goal"}
    assert "Allen" in client.get("/api/roster").get_json()["players"]


@pytest.mark.parametrize("body,status", [
    ({"name": "Allen", "clips": []}, 400),
    ({"name": "  ", "clips": ["DJI_20261017_0001/shot_001.mp4"]}, 400),
    ({"name": "Allen", "clips": ["DJI_20261017_0001/nope.mp4"]}, 404),
    ({"name": "Allen", "clips": ["../../etc/passwd"]}, 400),
])
def test_naming_rejects_bad_requests(env, body, status):
    client, base = env
    before = (base / "labels.json").read_text()
    assert client.post("/api/players/name", json=body).status_code == status
    assert (base / "labels.json").read_text() == before
