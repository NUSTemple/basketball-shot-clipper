"""AI pre-label scoring: rim-based geometry, within-video ranking, shot timing
inside merged clips, the score remap onto the UI's bins, which rows train the
model, and which model a job gets."""
import json
import random

import pytest

from shot_clipper import prelabel
from shot_clipper.clip_shots import PRELABEL_FILENAME


def test_event_times_match_how_cut_all_cuts_a_cluster():
    assert prelabel.event_times([30.0]) == [5.0]
    assert prelabel.event_times([30.0, 33.5]) == [5.0, 8.5]
    assert prelabel.event_times([2.0, 4.0]) == [2.0, 4.0]  # clip start clamped at 0


def test_net_roi_hangs_below_the_detected_rim():
    rim = (1000, 400, 1060, 420)  # 60px wide
    x1, y1, x2, y2 = prelabel.rim_net_roi(rim, 1920, 1080)
    assert y1 == 410 and y2 == int(420 + prelabel.NET_DEPTH_RIM_WIDTHS * 60)
    assert x1 < 1000 < 1060 < x2


def test_hoop_box_is_centred_on_the_rim_and_scaled():
    x1, y1, x2, y2 = prelabel.rim_hoop_box((1000, 400, 1060, 420), 1920, 1080)
    assert (x1 + x2) / 2 * 1920 == pytest.approx(1030)
    assert (x2 - x1) * 1920 == pytest.approx(60 * prelabel.HOOP_W_SCALE)


def test_ranks_ignore_a_camera_wide_offset():
    rows = [{"f": v} for v in (1.0, 3.0, 2.0)]
    shifted = [{"f": r["f"] * 100 + 7} for r in rows]  # same video, other camera's scale
    assert prelabel.rank_within_video(rows, ["f"]) == prelabel.rank_within_video(shifted, ["f"]) \
        == [[0.0], [1.0], [0.5]]


def test_remap_puts_model_cutoffs_on_ui_bins_and_keeps_order():
    lo, hi = 0.07, 0.31
    assert prelabel.remap(lo, lo, hi) == pytest.approx(prelabel.UI_MISS_BELOW)
    assert prelabel.remap(hi, lo, hi) == pytest.approx(prelabel.UI_MAKE_FROM)
    ys = [prelabel.remap(x / 100, lo, hi) for x in range(101)]
    assert ys == sorted(ys) and ys[0] == 0.0 and ys[-1] == pytest.approx(1.0)


def _synthetic(n_videos=4, per_video=40, seed=0):
    """Each video is a different 'camera': its net-motion values sit on its
    own scale and offset, so only the within-video order carries signal."""
    rnd = random.Random(seed)
    rows, labels, group_of = {}, {}, {}
    for v in range(n_videos):
        scale, offset = rnd.uniform(1, 20), rnd.uniform(-5, 5)
        for k in range(per_video):
            clip = f"v{v}/shot_{k:03d}.mp4"
            goal = k % 3 == 0
            n_shots = 2 if k % 5 == 0 else 1
            shots = []
            for e in range(n_shots):
                strong = goal and e == n_shots - 1  # in merged makes, the later shot went in
                signal = (0.8 if strong else 0.2) + rnd.uniform(-.15, .15)
                shots.append({"flow_coherence_mean": signal * scale + offset, "diff_peak": rnd.uniform(0, 1),
                              "event": e, "n_events": n_shots, "t_local": 5.0 + e})
            rows[clip], labels[clip], group_of[clip] = shots, ("goal" if goal else "no_goal"), f"v{v}"
    return rows, labels, group_of


def test_training_ranks_within_video_and_ignores_bookkeeping_fields():
    bundle, report = prelabel.train(*_synthetic())
    assert bundle["method"] == prelabel.METHOD
    assert set(bundle["features"]) == {"flow_coherence_mean", "diff_peak"}
    assert report["held_out_auc"] > 0.9  # despite every video having its own scale
    assert report["lo"] < report["hi"]
    assert report["held_out_makes_hidden"] <= prelabel.MAX_HIDDEN_MAKES * report["n_goals"]


def test_score_video_takes_each_clips_best_shot_and_skips_tiny_videos():
    rows, labels, group_of = _synthetic()
    bundle, _ = prelabel.train(rows, labels, group_of)
    video = {c.split("/")[1]: r for c, r in rows.items() if group_of[c] == "v0"}
    scores = prelabel.score_video(bundle, video)
    merged_goal = next(c.split("/")[1] for c in rows if group_of[c] == "v0" and len(rows[c]) > 1
                       and labels[c] == "goal")
    assert scores[merged_goal] >= prelabel.UI_MAKE_FROM
    tiny = dict(list(video.items())[:3])
    assert prelabel.score_video(bundle, tiny) == {}


def test_best_shots_points_at_the_shot_that_went_in():
    rows, labels, group_of = _synthetic()
    bundle, _ = prelabel.train(rows, labels, group_of)
    video = {c.split("/")[1]: r for c, r in rows.items() if group_of[c] == "v0"}
    best = prelabel.best_shots(bundle, video)
    merged_goals = [c.split("/")[1] for c in rows if group_of[c] == "v0" and len(rows[c]) > 1
                    and labels[c] == "goal"]
    assert merged_goals
    for clip in merged_goals:
        assert best[clip][1] == len(video[clip]) - 1  # the later shot is the make
    assert {c: s for c, (s, _) in best.items()} == prelabel.score_video(bundle, video)


def test_score_cut_clips_uses_each_clips_own_cluster_and_writes_the_file(tmp_path, monkeypatch):
    bundle, _ = prelabel.train(*_synthetic())
    calls = []

    def fake_features(path, rim, local_times, detector, device="cpu"):
        calls.append((path.name, local_times))
        strong = path.name in ("shot_002.mp4",)
        return [{"flow_coherence_mean": 0.9 if strong else random.random() * 0.3, "diff_peak": 0.5}
                for _ in local_times]

    monkeypatch.setattr(prelabel, "clip_shot_features", fake_features)
    makes = [10.0, 40.0, 42.0] + [80.0 + 20 * k for k in range(8)]  # clusters: [10], [40, 42], 8 singles
    cut = [(i, 0.0, tmp_path / f"shot_{i:03d}.mp4") for i in range(1, 11)]
    scores = prelabel.score_cut_clips(cut[::-1], makes, (0, 0, 1, 1), tmp_path, bundle, detector=None)
    assert dict(calls)["shot_002.mp4"] == [5.0, 7.0]
    assert dict(calls)["shot_001.mp4"] == [5.0]
    assert json.loads((tmp_path / PRELABEL_FILENAME).read_text()) == scores
    assert max(scores, key=scores.get) == "shot_002.mp4"


def test_hosted_jobs_use_their_own_override_then_the_global_model(tmp_path, monkeypatch):
    import joblib

    from shot_clipper.label_ui.app import user_slug

    monkeypatch.setattr(prelabel, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(prelabel, "MODEL_PATH", tmp_path / "local.joblib")
    ok = {"method": prelabel.METHOD}
    joblib.dump({**ok, "who": "global"}, tmp_path / prelabel.GLOBAL_MODEL_NAME)
    joblib.dump({**ok, "who": "anthony"}, tmp_path / f"{user_slug('anthony.ssdi@gmail.com')}.joblib")
    joblib.dump({**ok, "who": "local"}, tmp_path / "local.joblib")
    assert prelabel.account_slug("anthony.ssdi@gmail.com") == user_slug("anthony.ssdi@gmail.com")
    assert prelabel.model_for("anthony.ssdi@gmail.com")["who"] == "anthony"
    assert prelabel.model_for("someone.new@gmail.com")["who"] == "global"
    assert prelabel.model_for(None)["who"] == "local"


def test_a_model_from_another_feature_recipe_is_ignored(tmp_path):
    import joblib

    joblib.dump({"model": "old per-account model, no method tag"}, tmp_path / "old.joblib")
    joblib.dump({"method": "something-else"}, tmp_path / "other.joblib")
    assert prelabel.load_model(tmp_path / "old.joblib") is None
    assert prelabel.load_model(tmp_path / "other.joblib") is None
    assert prelabel.load_model(tmp_path / "missing.joblib") is None
