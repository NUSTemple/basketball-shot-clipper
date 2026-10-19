"""AI pre-label scoring: shot timing inside merged clips, the score remap
onto the UI's bins, which rows train the model, and that bookkeeping fields
never become features."""
import json
import random

import pytest

from shot_clipper import prelabel
from shot_clipper.clip_shots import PRELABEL_FILENAME


def test_event_times_match_how_cut_all_cuts_a_cluster():
    assert prelabel.event_times([30.0]) == [5.0]
    assert prelabel.event_times([30.0, 33.5]) == [5.0, 8.5]
    assert prelabel.event_times([2.0, 4.0]) == [2.0, 4.0]  # clip start clamped at 0


def test_remap_puts_model_cutoffs_on_ui_bins_and_keeps_order():
    lo, hi = 0.07, 0.31
    assert prelabel.remap(lo, lo, hi) == pytest.approx(prelabel.UI_MISS_BELOW)
    assert prelabel.remap(hi, lo, hi) == pytest.approx(prelabel.UI_MAKE_FROM)
    xs = [i / 100 for i in range(101)]
    ys = [prelabel.remap(x, lo, hi) for x in xs]
    assert ys == sorted(ys) and ys[0] == 0.0 and ys[-1] == pytest.approx(1.0)


def _synthetic(n_videos=3, per_video=40, seed=0):
    rnd = random.Random(seed)
    rows, labels, group_of = {}, {}, {}
    for v in range(n_videos):
        for k in range(per_video):
            clip = f"v{v}/shot_{k:03d}.mp4"
            goal = k % 3 == 0
            n_shots = 2 if k % 5 == 0 else 1
            shots = []
            for e in range(n_shots):
                strong = goal and e == n_shots - 1  # in merged makes, the later shot went in
                shots.append({"flow_coherence_mean": (0.8 if strong else 0.2) + rnd.uniform(-.15, .15),
                              "diff_peak": rnd.uniform(0, 1), "event": e, "n_events": n_shots, "t_local": 5.0 + e})
            rows[clip], labels[clip], group_of[clip] = shots, ("goal" if goal else "no_goal"), f"v{v}"
    return rows, labels, group_of


def test_training_ignores_bookkeeping_fields_and_scores_held_out_videos():
    bundle, report = prelabel.train(*_synthetic())
    assert set(bundle["features"]) == {"flow_coherence_mean", "diff_peak"}
    assert report["held_out_auc"] > 0.9
    assert report["lo"] < report["hi"]
    assert report["held_out_makes_hidden"] <= prelabel.MAX_HIDDEN_MAKES * report["n_goals"]


def test_merged_goal_clips_are_left_out_of_training_but_scored_by_best_shot(monkeypatch):
    rows, labels, group_of = _synthetic()
    seen = []
    real_fit = prelabel._fit

    def spy(rows_, labels_, clips_, features_):
        seen.extend(c for c in clips_ if not (len(rows_[c]) > 1 and labels_[c] == "goal"))
        return real_fit(rows_, labels_, clips_, features_)

    monkeypatch.setattr(prelabel, "_fit", spy)
    bundle, _ = prelabel.train(rows, labels, group_of)
    merged_goal = next(c for c in rows if len(rows[c]) > 1 and labels[c] == "goal")
    weak_first, strong_last = rows[merged_goal]
    assert prelabel.score_shots(bundle, [weak_first, strong_last]) >= prelabel.score_shots(bundle, [weak_first])


def test_score_cut_clips_uses_each_clips_own_cluster_and_writes_the_file(tmp_path, monkeypatch):
    bundle, _ = prelabel.train(*_synthetic())
    calls = []

    def fake_features(path, hoop, local_times, detector, device="cpu"):
        calls.append((path.name, local_times))
        return [{"flow_coherence_mean": 0.8, "diff_peak": 0.5}] * len(local_times)

    monkeypatch.setattr(prelabel, "clip_shot_features", fake_features)
    makes = [10.0, 40.0, 42.0, 80.0]  # clusters: [10], [40, 42], [80]
    cut = [(1, 10.0, tmp_path / "shot_001.mp4"), (2, 40.0, tmp_path / "shot_002.mp4"),
           (3, 80.0, tmp_path / "shot_003.mp4")]
    scores = prelabel.score_cut_clips(cut[::-1], makes, (0, 0, 1, 1), tmp_path, bundle, detector=None)
    assert dict(calls) == {"shot_001.mp4": [5.0], "shot_002.mp4": [5.0, 7.0], "shot_003.mp4": [5.0]}
    assert json.loads((tmp_path / PRELABEL_FILENAME).read_text()) == scores
    assert set(prelabel.bin_counts(scores)) == {"make", "unsure", "miss"}


def test_no_model_means_no_scoring(tmp_path):
    assert prelabel.load_model(tmp_path / "nope.joblib") is None
