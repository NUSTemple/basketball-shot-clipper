# Player identification: who scored

Goal: for each `goal`-labeled clip, know *who* made the shot - both so a
user can pull a single player's highlight reel, and so session stats
("makes by player") become possible later.

## Feasibility check (read this before building anything automated)

The source footage is **not** top-down drone video despite the `DJI_*`
filenames - it's a fixed sideline/corner camera at roughly hoop height,
shooting across a normal indoor court (3840x2160, confirmed via `ffprobe`
against `DJI_20260725152636_0001_D.MP4`). A sample frame pulled with
`ffmpeg -ss 20 -frames:v 1`:

- **Jersey numbers are clearly legible** on players near/at mid-range
  from the camera ("20", "7" both readable at full frame resolution).
- **Faces are small and usually turned away.** Players face the hoop, not
  the camera, for the entire shot motion that matters (the few seconds
  before release) - the moments a face is frontal and identifiable are
  incidental (walking past camera, standing around), not the scoring
  moment itself.

This flips the naive assumption: **face recognition is the weaker signal
here, jersey-number OCR is the stronger one.** Any design should treat
face matching as a fallback, not the primary mechanism, and both as
suggestions a human confirms - never an auto-applied label. That mirrors
the existing shot filter's own philosophy (README, "Improving precision
with a trained filter": "a triage aid, not an auto-filter").

## Staged plan

**Phase 1 - manual tagging (shipped on this branch).** "Scorer" and
"Assist" pickers next to the video in the Review panel - a search box
with matching roster names as clickable pills below it (checkmark = the
current pick, "+" = click to pick, a typed unmatched name becomes its own
"+" pill to add and tag in one click) - backed by a small roster
(`data/dataset/roster.json` via `src/shot_clipper/roster.py`) and
`scorer`/`assist` fields alongside `label`/`stars` in
`data/dataset/labels.json` (`/api/scorer`, `/api/assist`, `/api/roster` in
`label_ui/app.py`). This alone answers "who scored" today, at the cost of
one search-and-click per goal clip, and it's the ground truth any
automated suggester in Phase 2/3 would need to be evaluated against - same
reason `shot-clipper-train-filter` needed hand-labeled clips before a
trained filter was possible at all.

`shot-clipper-build-dataset --player NAME` and `/api/export-clips` both
already respect the `scorer` tag (filename suffix, e.g.
`DJI_0001_D__shot_012_5star_Alice.mp4`), so a per-player highlight reel
is just `--player Alice --min-stars 4` once enough clips are tagged.

**Phase 2 - jersey-number OCR suggestion (not built).** Per candidate
clip: run a person detector (YOLO's `person` class - already a model
family this repo depends on) over the frames around the shot event,
crop each detected player, OCR the jersey digits (candidates: EasyOCR,
PaddleOCR, or a small custom-trained digit detector - regular OCR models
are tuned for document text, not curved cloth numbers at a distance, so
whichever is picked needs validating on real frames before trusting it).
Map the recognized number to a name via a `roster.json` extended with a
`{name, jersey_number}` pairing, and surface it as a **suggested** scorer
(pre-selected in the dropdown, clearly marked as unconfirmed - like
`filter_score` today) rather than silently writing `labels.json`.

Known failure modes to validate against before shipping: motion blur
during the shot motion itself, jersey number occluded by the ball/arm
mid-shot, two players with the same or partially-obscured number,
scrimmages with no numbers at all (bibs/vests only, or plain shirts).

**Phase 2b - clothing/appearance clustering (prototyped, not built).**
The idea: track each player through a whole session with YOLO's built-in
tracker (`model.track()`, ByteTrack - no new dependency), cluster tracks
by appearance into ~10-15 per-session identities, and label each cluster
once instead of tagging every goal clip individually - far less manual
work than Phase 1 alone once a session has many goals. This is worth
more than Phase 3's face fallback for this footage specifically: the
sample frame showed players in visually distinct casual clothing (not
matched team kits), not just distinct jersey numbers.

A quick validation (person detection + ByteTrack + per-track mean HSV
histogram of the torso region, clustered with scipy hierarchical
clustering, 90s window of one video) showed the idea is directionally
sound but a plain color histogram alone isn't enough to ship:

- ByteTrack fragments identities constantly under fast motion and
  occlusion - 30 track IDs came out of a 90s window that has at most
  ~10 real people in it.
- The histogram *does* carry signal - fragments of the same real person
  correlate ~0.85-0.97, different people ~0.3-0.6 - so clustering the
  fragments back together is the right idea.
- But clustering purely on that correlation, with no other constraint,
  produced both correct merges (two fragments of a pink-shirted player
  reunited) and clear mistakes (a 12-track cluster mixing several
  different people who all happen to wear dark colors under gym
  lighting).
- The clustering also ignored a free, decisive constraint: two tracks
  that overlap in time cannot be the same person. Not enforcing that is
  most of why the dark-clothing cluster is such a mess - a real
  implementation should build this in as a must-not-link constraint
  before clustering, not treat it as a nice-to-have.

Building this for real would need: (1) the temporal must-not-link
constraint above, (2) a real person re-ID embedding instead of a raw
color histogram (a small pretrained re-ID model is far more robust to
lighting/pose than color alone, and would tell apart two players in
similar dark clothing that color cannot), and (3) combining with Phase
2's jersey-number OCR wherever a number is legible, since the two
signals fail in different situations (OCR fails on motion blur/occluded
numbers; appearance fails on similar-colored clothing) - the same
"combine complementary weak signals" lesson the shot filter itself
already validated (README: trajectory + net-motion features roughly
double how many false positives can be dropped together vs. either
alone).

**Phase 3 - face-embedding fallback (not built, lower priority given the
feasibility finding above).** For clips where no jersey number is
readable, fall back to a face embedding (e.g. a lightweight ArcFace-style
model) matched against faces enrolled once per player (a short "look at
the camera" clip, or manually cropped frames from Library). Given how
rarely a frontal face is available at the moment of a shot, this is
realistically a **cross-clip identity hint** (recognize the same player
elsewhere in the session where they do face the camera, then propagate
via person tracking through the shot) rather than a per-shot face match -
plain per-shot face recognition will mostly come back empty.

## Architecture sketch

A single entry point that Phase 2/3 would implement, kept independent of
the UI so either can ship without the other:

```python
# src/shot_clipper/person_id/suggest.py  (interface only for now)

@dataclass
class ScorerSuggestion:
    name: str
    confidence: float
    method: str  # "jersey_ocr" | "face_embedding"

def suggest_scorer(clip_path: Path, hoop_bbox_norm, shot_time_sec: float,
                    roster: list[dict]) -> list[ScorerSuggestion]:
    """Ranked suggestions for who scored, highest confidence first. Never
    writes labels.json directly - the label UI shows the top suggestion
    pre-selected but unconfirmed, same pattern as filter_score."""
```

`label_ui/app.py` would call this from `/api/clips` (best-effort, cached
per clip like `_filter_scores.json`) and the Review panel would render it
as a highlighted-but-not-yet-saved pill in the existing scorer/assist
picker (`pillsHtml`/`scorerRowHtml` in `templates/index.html`) - no new UI
surface needed, just a `suggested: true` flag on the pill already there.

## Why not build Phase 2/2b/3 now

All three need a labeled evaluation set to know if they're actually any
good (same lesson as the shot filter: a hand-tuned rule "looked right" on
one video and was 34% precision on nine). Phase 1's manual tags are that
evaluation set in progress - worth letting it accumulate before investing
in OCR/embedding model selection and tuning. Phase 2b's quick prototype
(above) is a cheap way to de-risk the idea early, not a substitute for
that - it confirmed the general direction but also surfaced real gaps
(must-not-link constraints, a better embedding) that need solving before
it could feed real suggestions into the UI.
