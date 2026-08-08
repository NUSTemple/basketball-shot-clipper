"""Automated "who scored" suggestion - see docs/PLAYER_IDENTIFICATION.md.

Not implemented yet: manual tagging (roster.py, the Review panel's scorer
picker) is the shipped mechanism today. This package is the interface a
jersey-OCR or face-embedding suggester would implement later, kept
separate so the UI integration didn't have to wait on either.
"""
from .suggest import ScorerSuggestion, suggest_scorer

__all__ = ["ScorerSuggestion", "suggest_scorer"]
