"""Score how highlight-worthy each goal clip is with a local vision-language
model (Qwen3-VL via MLX-VLM on Apple Silicon). Part of shot-clipper-highlights.

MLX-VLM lives in its own venv, so this file imports nothing from the
package - highlights.py runs it as a script:

    ~/.venvs/mlx-vlm/bin/python -I highlight_vlm.py <jobs.json> <out.jsonl>

jobs.json is a list of {"clip": key, "path": mp4, "hoop": [x1,y1,x2,y2]
normalized, "make": clip-local seconds of the make}. Frames are taken around
each clip's own make time (a merged clip's make isn't at 5s). One JSON line
per clip is appended to out.jsonl; clips already there are skipped, so a run
can be stopped and resumed.
"""
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# MoE, ~3B active parameters per token: ~13s a clip on an M3 Max. Against the
# owner's stars (141 goals, 17 Oct 2026) its highlight_score had Spearman 0.55
# and its top 23 held no 1-star clip; the dense 32B was ~6 min a clip.
MODEL = "mlx-community/Qwen3-VL-30B-A3B-Instruct-8bit"
WIDE_OFFSETS = [-3.5, -2.0, -1.0, -0.4, 0.2, 1.2]  # seconds from the make
ZOOM_OFFSETS = [-0.4, 0.0, 0.6]
WIDE_WIDTH = 896
ZOOM_WIDTH = 576

PROMPT = """These frames come from a fixed wide-angle camera filming amateur basketball \
(pickup games or shoot-arounds). Several people may be shooting at the same time.
The first {n_wide} images are the whole court at t = {wide_times} seconds. \
The last {n_zoom} images are a zoom on the area around the hoop at t = {zoom_times} seconds.
A shot goes in at about t = {make_at}s. Judge THAT shot only.

Rate how highlight-worthy it is - would someone keep it in a highlight reel? \
Things that make it better: a long or difficult shot, a defender contesting it, \
an athletic or creative move (crossover, spin, euro step, reverse layup, \
fadeaway, and-one, dunk), a live game rather than a warm-up, a clean swish, \
teammates reacting. Things that make it worse: an uncontested warm-up shot, \
a routine layup with nobody around, the shot being hard to see.

Reply with ONLY this JSON, no other text:
{{"shot_type": "layup|reverse_layup|dunk|floater|hook|jump_shot|fadeaway|three_pointer|putback|other",
 "distance": "at_rim|short|mid_range|three_point",
 "defense": "none|light|contested",
 "game_context": "warm_up|live_game|unclear",
 "finish": "swish|rim_in|bank|unclear",
 "difficulty": 1-5,
 "athleticism": 1-5,
 "reaction": 1-5,
 "highlight_score": 1-10,
 "description": "at most 15 words about the play"}}"""


def grab_frame(clip: str, t: float, out: Path, vf: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, t):.2f}", "-i", clip,
                    "-frames:v", "1", "-vf", vf, "-q:v", "3", str(out)], check=True)


def zoom_filter(hoop) -> str:
    """A region around the hoop wide enough to see a shooter near the basket
    and tall enough to reach the floor below it."""
    x1, y1, x2, _ = hoop
    w, h = 0.30, 0.50
    left = min(max((x1 + x2) / 2 - w / 2, 0.0), 1.0 - w)
    top = min(max(y1 - 0.10, 0.0), 1.0 - h)
    return f"crop=iw*{w}:ih*{h}:iw*{left:.4f}:ih*{top:.4f},scale={ZOOM_WIDTH}:-2"


def frame_times(make: float) -> tuple[list[float], list[float]]:
    return ([round(max(0.0, make + o), 2) for o in WIDE_OFFSETS],
            [round(max(0.0, make + o), 2) for o in ZOOM_OFFSETS])


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main() -> None:
    jobs_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    jobs = json.loads(jobs_path.read_text())
    done = set()
    if out_path.exists():
        done = {json.loads(line)["clip"] for line in out_path.read_text().splitlines() if line}
    todo = [j for j in jobs if j["clip"] not in done]
    print(f"{len(todo)} clips to score ({len(done)} already done)", flush=True)
    if not todo:
        return

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(MODEL)
    n_images = len(WIDE_OFFSETS) + len(ZOOM_OFFSETS)
    for n, job in enumerate(todo, start=1):
        t0 = time.time()
        wide, zoom = frame_times(job["make"])
        prompt = apply_chat_template(processor, model.config, PROMPT.format(
            n_wide=len(wide), n_zoom=len(zoom), make_at=round(job["make"], 1),
            wide_times=", ".join(f"{t:g}" for t in wide), zoom_times=", ".join(f"{t:g}" for t in zoom)),
            num_images=n_images)
        with tempfile.TemporaryDirectory() as tmp:
            images = []
            for i, t in enumerate(wide):
                p = Path(tmp) / f"w{i:02d}.jpg"
                grab_frame(job["path"], t, p, f"scale={WIDE_WIDTH}:-2")
                images.append(str(p))
            for i, t in enumerate(zoom):
                p = Path(tmp) / f"z{i:02d}.jpg"
                grab_frame(job["path"], t, p, zoom_filter(job["hoop"]))
                images.append(str(p))
            result = generate(model, processor, prompt, image=images,
                              max_tokens=250, temperature=0.0, verbose=False)
        text = getattr(result, "text", result)
        parsed = parse_json(text)
        row = {"clip": job["clip"], "seconds": round(time.time() - t0, 1), "vlm": parsed,
               "raw": None if parsed else text}
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        score = parsed.get("highlight_score") if parsed else "PARSE-FAIL"
        print(f"[{n}/{len(todo)}] {job['clip']} vlm={score} ({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
