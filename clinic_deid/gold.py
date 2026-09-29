"""Load synthetic notes and resolve gold-label spans to character offsets.

Gold labels are stored as (label, exact span text) pairs so the file stays
human-readable; this module converts them to (start, end, label) spans.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel

from .detectors import Span


class PhiGold(BaseModel):
    label: str
    text: str
    occurrence: int = 1


class KeyFact(BaseModel):
    fact: str
    keywords: List[str]


class NoteGold(BaseModel):
    note_id: str
    phi: List[PhiGold]
    reference_summary: str
    key_facts: List[KeyFact]


class Note(BaseModel):
    id: str
    text: str


def load_notes(path: str | Path) -> List[Note]:
    with open(path) as f:
        return [Note(**n) for n in json.load(f)]


def load_gold(path: str | Path) -> Dict[str, NoteGold]:
    with open(path) as f:
        return {g["note_id"]: NoteGold(**g) for g in json.load(f)}


def resolve_spans(note_text: str, phi_golds: List[PhiGold]) -> List[Span]:
    """Map each gold (label, text) pair to its character offsets."""
    spans: List[Span] = []
    for g in phi_golds:
        start = 0
        idx = -1
        for _ in range(g.occurrence):
            idx = note_text.find(g.text, start)
            if idx == -1:
                raise ValueError(
                    f"Gold span text not found (occurrence {g.occurrence}): "
                    f"{g.text!r}")
            start = idx + 1
        spans.append(Span(idx, idx + len(g.text), g.label, g.text))
    spans.sort(key=lambda s: s.start)
    return spans
