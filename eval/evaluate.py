"""Evaluate de-identification + summarization against synthetic gold labels.

De-id: entity-level precision/recall/F1 per PHI category on exact
(start, end, label) span matches.
Summarization: ROUGE-1 F1 vs the synthetic reference summary, plus the
fraction of key clinical facts whose keywords appear in the summary.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from clinic_deid import (PHI_LABELS, detect, load_gold, load_notes,
                         redact, resolve_spans, summarize)


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def rouge1_f1(pred: str, ref: str) -> float:
    p, r = Counter(tokenize(pred)), Counter(tokenize(ref))
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    prec = overlap / sum(p.values())
    rec = overlap / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def fact_coverage(summary_text: str, key_facts) -> float:
    low = summary_text.lower()
    if not key_facts:
        return 1.0
    hit = sum(1 for f in key_facts
              if any(k.lower() in low for k in f.keywords))
    return hit / len(key_facts)


def evaluate(notes_path: Path, gold_path: Path) -> dict:
    notes = load_notes(notes_path)
    gold = load_gold(gold_path)

    tp = Counter(); fp = Counter(); fn = Counter()
    rouge_scores, coverages = [], []

    for note in notes:
        g = gold[note.id]
        pred_keys = {s.key() for s in detect(note.text)}
        gold_keys = {s.key() for s in resolve_spans(note.text, g.phi)}
        for key in pred_keys & gold_keys:
            tp[key[2]] += 1
        for key in pred_keys - gold_keys:
            fp[key[2]] += 1
        for key in gold_keys - pred_keys:
            fn[key[2]] += 1

        summary = summarize(redact(note.text))["summary_text"]
        rouge_scores.append(rouge1_f1(summary, g.reference_summary))
        coverages.append(fact_coverage(summary, g.key_facts))

    rows = []
    for label in PHI_LABELS:
        p = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 1.0
        r = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 1.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"label": label, "tp": tp[label], "fp": fp[label],
                     "fn": fn[label], "precision": p, "recall": r, "f1": f1})

    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    micro_p = TP / (TP + FP) if TP + FP else 1.0
    micro_r = TP / (TP + FN) if TP + FN else 1.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0

    return {
        "per_category": rows,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "summary": {
            "rouge1_f1_mean": float(np.mean(rouge_scores)),
            "fact_coverage_mean": float(np.mean(coverages)),
            "per_note_rouge1": [float(s) for s in rouge_scores],
            "per_note_coverage": [float(s) for s in coverages],
        },
    }


def main() -> None:
    report = evaluate(REPO / "data" / "notes.json",
                      REPO / "data" / "gold_labels.json")
    print(f"{'PHI category':<10}{'TP':>5}{'FP':>5}{'FN':>5}"
          f"{'Prec':>8}{'Rec':>8}{'F1':>8}")
    for row in report["per_category"]:
        print(f"{row['label']:<10}{row['tp']:>5}{row['fp']:>5}{row['fn']:>5}"
              f"{row['precision']:>8.3f}{row['recall']:>8.3f}{row['f1']:>8.3f}")
    m = report["micro"]
    print(f"\nMicro-averaged de-id: P={m['precision']:.3f} "
          f"R={m['recall']:.3f} F1={m['f1']:.3f}")
    s = report["summary"]
    print(f"Summary ROUGE-1 F1 (mean): {s['rouge1_f1_mean']:.3f}")
    print(f"Key-fact coverage (mean):  {s['fact_coverage_mean']:.3f}")

    out = REPO / "eval" / "report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nFull report written to {out}")


if __name__ == "__main__":
    main()
