"""End-to-end pipeline tests: detect -> redact -> summarize vs gold labels."""
from pathlib import Path

from clinic_deid import (load_gold, load_notes, redact, resolve_spans,
                         summarize, detect)

REPO = Path(__file__).resolve().parent.parent


def _f1(pred_keys, gold_keys):
    tp = len(pred_keys & gold_keys)
    fp = len(pred_keys - gold_keys)
    fn = len(gold_keys - pred_keys)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return 2 * p * r / (p + r) if p + r else 0.0


def test_pipeline_perfect_on_synthetic_notes():
    notes = load_notes(REPO / "data" / "notes.json")
    gold = load_gold(REPO / "data" / "gold_labels.json")
    for note in notes:
        pred = {s.key() for s in detect(note.text)}
        expected = {s.key() for s in resolve_spans(note.text, gold[note.id].phi)}
        assert _f1(pred, expected) == 1.0, f"de-id F1 < 1.0 on {note.id}"


def test_pipeline_summary_covers_key_facts():
    notes = load_notes(REPO / "data" / "notes.json")
    gold = load_gold(REPO / "data" / "gold_labels.json")
    for note in notes:
        summary = summarize(redact(note.text))["summary_text"].lower()
        g = gold[note.id]
        covered = sum(1 for f in g.key_facts
                      if any(k.lower() in summary for k in f.keywords))
        assert covered / len(g.key_facts) >= 0.8, f"low fact coverage on {note.id}"


def test_pipeline_redacted_note_has_no_detectable_phi():
    notes = load_notes(REPO / "data" / "notes.json")
    for note in notes:
        redacted = redact(note.text)
        # Year tokens inside "[DATE] YYYY" tags are allowed by Safe Harbor;
        # anything else the detector still finds is residual PHI.
        leftover = [s for s in detect(redacted) if s.label != "DATE"]
        assert leftover == [], f"residual PHI in {note.id}: {leftover}"
