# Clinical Notes De-identification + Summarization (HIPAA Safe Harbor)
[![CI](https://github.com/akhilreddyb23/clinical-notes-deid-summarization/actions/workflows/ci.yml/badge.svg)](https://github.com/akhilreddyb23/clinical-notes-deid-summarization/actions)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


A portfolio demonstration of two classic clinical-NLP techniques, running
entirely on CPU with no model downloads:

1. **De-identification** per the HIPAA Safe Harbor method — regex and
   rule-based detectors find names, dates (year preserved), phone/fax
   numbers, emails, SSNs, MRNs, account numbers, street/city/ZIP addresses,
   and ages over 89, then replace them with tags like `[NAME]`, `[DATE] 1957`.
2. **Extractive summarization** — sentences are scored on clinical cue words
   (complaint / finding / plan vocabulary plus section-header boosts) and
   rendered as a structured summary: chief complaint, key findings,
   assessment & plan.

All patient notes in `data/` are **synthetic and fictitious**, written for
this demo. No real PHI, no real patients, no credentials anywhere.

## Architecture

```
                        +-------------------+
                        | data/notes.json   |  synthetic notes (fictitious)
                        +---------+---------+
                                  |
              +-------------------v-------------------+
              | clinic_deid/detectors.py              |
              | regex detectors: NAME DATE PHONE      |
              | EMAIL SSN MRN ACCOUNT ADDRESS AGE     |
              | (priority-ordered overlap resolution) |
              +--------+--------------+---------------+
                       | spans        |
        +--------------v---+    +-----v--------------+
        | gold_labels.json |    | redact.py          |
        | span-level gold  |    | [NAME] [DATE] ...  |
        | (label + text)   |    +-----+--------------+
        +--------------+---+          |
                       |        +-----v--------------+
                       |        | summarize.py       |
                       |        | extractive, CPU    |
                       |        +-----+--------------+
                       |              |
              +--------v--------------v---------+
              | eval/evaluate.py                |
              | entity P/R/F1 per PHI category  |
              | ROUGE-1 + key-fact coverage     |
              +---------------------------------+
```

## Install

Requires Python 3.10+. CPU-only dependencies (numpy, pydantic, pytest).

```bash
cd clinical-notes-deid-summarization
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run the demo

De-identify and summarize the synthetic notes:

```bash
.venv/bin/python demo.py                    # all notes, pretty output
.venv/bin/python demo.py --note-id note_01  # a single note
.venv/bin/python demo.py --json             # machine-readable output
.venv/bin/python demo.py --eval             # run the evaluation report
```

## Run the tests

```bash
.venv/bin/python -m pytest tests/ -q
```

47 tests cover every PHI detector category, redaction behavior (tags,
year preservation on dates, ages ≤ 89 untouched), the summarizer
(sections, key-term capture, empty input), and the end-to-end pipeline
against the gold labels.

## What the eval shows

`eval/evaluate.py` (also via `demo.py --eval`) reports, on the three
synthetic notes:

- **De-identification:** entity-level precision/recall/F1 per PHI category
  on exact `(start, end, label)` span matches — micro-averaged F1 **1.000**
  on the synthetic set.
- **Summarization:** ROUGE-1 F1 ≈ **0.51** against the synthetic reference
  summaries, and key clinical-fact keyword coverage ≈ **0.82**.

These numbers measure the demo on its own synthetic data; they are not
claims about performance on real clinical text, which would need a proper
annotated corpus and review before any real-world use.
