import random

from shot_clipper import highlights


def _goals(n):
    return [{"key": f"V/shot_{i:03d}.mp4", "video": "V", "clip": f"shot_{i:03d}.mp4", "make": 5.0,
             "score": 0.9, "ai_score": float(n - i)} for i in range(n)]


def test_stars_follow_rank_shares_and_every_goal_gets_one():
    ranked = _goals(20)
    highlights.assign_stars(ranked)
    stars = [g["stars"] for g in ranked]
    assert stars == sorted(stars, reverse=True)
    assert stars.count(5) == 3 and stars.count(4) == 6
    tiny = _goals(1)
    highlights.assign_stars(tiny)
    assert tiny[0]["stars"] == 5


def test_effects_never_put_two_slow_motions_back_to_back():
    effects = highlights.pick_effects(500, {"plain": 1, "replay": 10, "speed_ramp": 10}, random.Random(0))
    assert all(not (a in highlights.SLOW_EFFECTS and b in highlights.SLOW_EFFECTS)
               for a, b in zip(effects, effects[1:]))


def test_ai_score_prefers_the_vlm_and_penalises_warm_ups():
    goal = {"score": 0.9}
    live = {"highlight_score": 6, "difficulty": 3, "athleticism": 3, "game_context": "live_game"}
    warm = {**live, "game_context": "warm_up"}
    assert highlights.ai_score(goal, live) > highlights.ai_score(goal, warm)
    assert highlights.ai_score(goal, {"highlight_score": "n/a"}) > 0  # junk from the model doesn't crash
    assert highlights.ai_score({"score": 0.95}, None) > highlights.ai_score({"score": 0.7}, None)


def test_three_variants_differ_and_countdown_ends_on_the_best_play():
    ranked = _goals(30)
    highlights.assign_stars(ranked)
    videos = {"V": {"rim_px": [960, 300, 1000, 320], "frame": [1920, 1080]}}
    title = ("SATURDAY RUN", "17 Oct 2026")
    top = highlights.build_variant("top_plays", ranked, videos, title, 1)
    count = highlights.build_variant("countdown", ranked, videos, title, 2)
    fast = highlights.build_variant("fast_cut", ranked, videos, title, 3)
    files = lambda p: [c["file"] for c in p["clips"]]  # noqa: E731
    assert len(files(top)) == 20 and files(top) == sorted(files(top))  # in the order they happened
    assert files(count)[-1] == "reels/V/shot_000.mp4" and count["clips"][-1]["effect"] == "zoom_replay"
    assert count["titleLines"][0] == "TOP 10"
    assert not any(c["effect"] in highlights.SLOW_EFFECTS for c in fast["clips"])
    assert len({tuple(files(top)), tuple(files(count)), tuple(files(fast))}) == 3
    assert top["clips"][0]["rimX"] == 51.0
    assert highlights.build_variant("top_plays", [], videos, title, 1) is None


def test_session_title_comes_from_the_dji_file_date(tmp_path):
    folder = tmp_path / "DJI_001"
    folder.mkdir()
    (folder / "DJI_20261017170711_0001_D.MP4").touch()
    assert highlights.session_title(folder) == ("SATURDAY RUN", "17 Oct 2026")
    other = tmp_path / "phone"
    other.mkdir()
    assert highlights.session_title(other)[0] == "HIGHLIGHTS"
