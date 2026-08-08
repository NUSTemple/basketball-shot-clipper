"""Interface for an automated "who scored" suggester.

See docs/PLAYER_IDENTIFICATION.md for the full design and the feasibility
finding behind it: this footage is a fixed sideline camera, not top-down
drone video, so jersey numbers are legible but faces are mostly turned
away - a jersey-OCR implementation is the one worth building first, with
face-embedding matching as a fallback for clips with no readable number.

Neither backend is implemented yet. `suggest_scorer` is the contract
`label_ui/app.py` would call from `/api/clips` once one exists: a ranked,
best-effort list of candidates that the Review panel shows as a
pre-selected but unconfirmed option (like `filter_score` today) - never
written to labels.json without a human confirming it, same reasoning as
the shot filter being "a triage aid, not an auto-filter" (README).
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScorerSuggestion:
    name: str
    confidence: float  # 0..1, not comparable across methods
    method: str  # "jersey_ocr" | "face_embedding"


def suggest_scorer(clip_path: Path, hoop_bbox_norm, shot_time_sec: float,
                    roster: list[dict]) -> list[ScorerSuggestion]:
    """Ranked suggestions for who scored in `clip_path`, highest confidence
    first (empty if nothing usable was found - e.g. no legible jersey
    number and no face match).

    roster: [{"name": str, "jersey_number": int | None}, ...] - the
    extended roster schema Phase 2 needs (today's roster.json is just
    names; jersey numbers aren't collected yet).

    Raises NotImplementedError - no jersey-OCR or face-embedding backend
    has been picked or validated against real clips yet (see "Why not
    build Phase 2/3 now" in the design doc). Implement by dispatching to
    a jersey_ocr module and/or a face_embedding module and merging their
    suggestions, not by growing this function directly.
    """
    raise NotImplementedError(
        "no automated scorer suggester is implemented yet - see "
        "docs/PLAYER_IDENTIFICATION.md. Tag scorers manually via the "
        "Review panel (roster.py) in the meantime."
    )
