#!/usr/bin/env python3
"""Demo CLI: de-identify + summarize the synthetic clinical notes.

Usage:
    python demo.py                 # process all synthetic notes
    python demo.py --note-id note_01
    python demo.py --eval          # run the evaluation report
    python demo.py --json          # machine-readable output
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from clinic_deid import load_notes, redact_with_spans, summarize

BANNER = ("NOTE: all patient data below is SYNTHETIC and fictitious, "
          "generated for this demo. No real PHI is involved.")


def run_note(note, as_json: bool):
    redacted, spans = redact_with_spans(note.text)
    summary = summarize(redacted)
    if as_json:
        print(json.dumps({
            "note_id": note.id,
            "spans": [{"start": s.start, "end": s.end, "label": s.label,
                       "text": s.text} for s in spans],
            "redacted": redacted,
            "summary": summary,
        }, indent=2))
        return
    print("=" * 72)
    print(f"NOTE {note.id}  —  {BANNER}")
    print("=" * 72)
    print(f"\nDetected {len(spans)} PHI spans:")
    for s in spans:
        print(f"  [{s.label:>7}] {s.text!r}")
    print("\n--- DE-IDENTIFIED NOTE ---")
    print(redacted)
    print("\n--- GENERATED SUMMARY ---")
    print(summary["summary_text"])
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Clinical notes de-id + summarization demo")
    ap.add_argument("--note-id", default=None, help="process one note, e.g. note_01")
    ap.add_argument("--eval", action="store_true", help="run eval/evaluate.py")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="emit JSON instead of pretty text")
    args = ap.parse_args()

    if args.eval:
        sys.path.insert(0, str(REPO / "eval"))
        from evaluate import main as eval_main
        eval_main()
        return

    notes = load_notes(REPO / "data" / "notes.json")
    if args.note_id:
        notes = [n for n in notes if n.id == args.note_id]
        if not notes:
            sys.exit(f"unknown note id: {args.note_id}")
    for note in notes:
        run_note(note, args.as_json)


if __name__ == "__main__":
    main()
