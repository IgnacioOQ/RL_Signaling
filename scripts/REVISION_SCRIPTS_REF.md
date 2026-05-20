---
status: active
type: reference
id: manuscript.revision_scripts_ref
description: CLI reference for the RL_Signaling cross-paper revision-tooling scripts — response_align.py (artifact drift checker), word_count.py (texcount wrapper), dash_audit.py (em-dash counter), bib_unused.py (bibliography hygiene) — covering flags, exit-code policy, JSON output, and per-phase invocation order for a revise-and-resubmit cycle.
label: [python, agent]
injection: informational
volatility: stable
scope: general
repository: [RL_Signaling]
last_checked: '2026-05-19'
---

# Revision Tooling Scripts Reference

The RL_Signaling repository carries four standalone Python scripts under `scripts/` that audit a paper revise-and-resubmit cycle. This document is the lookup reference for them: it lists every command-line flag, the exit-code policy, the JSON output mode, and the order in which each script should run across the phases of a revision. Reach for it when wiring the scripts into a session, a checklist, or CI, or when copying the toolkit into another paper's repository — the scripts take file paths as arguments and carry no hardcoded coupling to any single manuscript. The companion `scripts/README.md` is a shorter quick-start; this document is the complete specification.

## Overview

A paper revision keeps three documents in sync — an operational checklist of atomic `.tex` edits, a formal response narrative addressed to reviewers, and the manuscript source. These drift apart silently. The toolkit makes that drift visible and tracks two other revision metrics (word count, em-dash density) plus bibliography hygiene.

| Script | Purpose | Typical trigger |
|:---|:---|:---|
| `response_align.py` | Drift checker across the operational checklist, the formal response narrative, and the manuscript `.tex`. | After every revision round; as a pre-submission final check. |
| `word_count.py` | `texcount` wrapper reporting body / header / caption counts, with target comparison and per-section breakdown. | During a word-count-reduction pass against a journal limit. |
| `dash_audit.py` | Em-dash counter — LaTeX `---` and Unicode `—`, ignoring en-dashes `--`. | During an em-dash sweep or voice-revision pass. |
| `bib_unused.py` | Bibliography hygiene — entries defined but never cited, and (with `--reverse`) cite keys with no bib entry. | Pre-submission bibliography trim. |

## Shared conventions

All four scripts share the following surface.

- **Python**: 3.10+ floor. No third-party dependencies (the standard library only; `pytest` is used by the test file but not by the scripts).
- **`--help`**: every script prints full usage via `argparse`.
- **`--json`**: emit a machine-readable JSON object instead of the human-readable text report. Useful for diffing across revisions or wiring into CI.
- **`--soft`**: force exit code 0 even when drift is detected. Default behaviour is exit-1 on drift so the scripts can gate a CI job or pre-commit hook.
- **Standalone**: no shared internal module. Each `.py` file can be copied into another paper's repository individually.
- **Path arguments**: all input files are passed as CLI arguments; nothing is hardcoded.

### Exit-code policy

| Code | Meaning |
|:---|:---|
| `0` | No drift / under target / audit clean — or `--soft` was passed. |
| `1` | Drift detected (see each script for its drift definition). |
| `2` | Usage error — a required input file was not found, or no input matched. |

## response_align.py

Catches drift between the three coupled revision artifacts. Runs four independent checks and reports each as its own section.

- **Check A** — checklist items marked `[x]` with no matching entry in the response narrative.
- **Check B** — response-narrative entries with no matching checklist item.
- **Check C** — italic verbatim quotes (`*"..."*`) in the narrative that no longer appear in the manuscript `.tex`. Quotes are classified as pre-revision (deleted wording, in a "previous *X* — has been replaced with *Y*" pattern) or post-revision; only post-revision quotes are checked, unless `--strict-quotes` is passed.
- **Check D** — `[file.tex:NNN]` line anchors in checklist sub-bullets whose target line no longer matches the documented edit.

### response_align.py synopsis

```bash
python scripts/response_align.py \
  --checklist  "manuscript/reviewers/responses_checklist.md" \
  --responses  "manuscript/reviewers/responses_to_reviewers.md" \
  --manuscript manuscript/main_v2.tex manuscript/Appendix.tex
```

### response_align.py options

| Flag | Required | Meaning |
|:---|:---|:---|
| `--checklist PATH` | yes | Operational checklist markdown file. |
| `--responses PATH` | yes | Formal response narrative markdown file. |
| `--manuscript PATH [PATH ...]` | yes | One or more `.tex` source files to validate quotes and anchors against. |
| `--strict-quotes` | no | Check every italic verbatim quote, including pre-revision deleted wording (default: skip pre-revision quotes). |
| `--json` | no | Emit JSON instead of the text report. |
| `--soft` | no | Force exit 0 even on drift. |

Exit `1` if any of the four checks has hits, otherwise `0`.

## word_count.py

Wraps `texcount` and reports word counts, optionally compared against a journal limit and broken down per section.

### word_count.py synopsis

```bash
python scripts/word_count.py manuscript/main_v2.tex --target 9000 --per-section
```

### word_count.py options

| Flag | Required | Meaning |
|:---|:---|:---|
| `tex_file` | yes | Positional — the `.tex` source to count. |
| `--target N` | no | Journal word limit. Reports body-vs-target margin and exits `1` when over. |
| `--per-section` | no | Include a per-section breakdown (one row per `\section` / `\subsection`). |
| `--json` | no | Emit JSON instead of the text report. |
| `--soft` | no | Force exit 0 even when over target. |

Reports `Sum count`, `body` (words in text), `headers`, and `captions`. Exit `1` when `--target` is set and the body count exceeds it; `2` if the file is not found; otherwise `0`. Requires `texcount` on `PATH` (ships with TeX Live / MacTeX).

## dash_audit.py

Counts em-dash occurrences in `.tex` and `.md` files. Distinguishes the LaTeX em-dash `---` (three hyphens) from the Unicode em-dash `—`, and ignores the LaTeX en-dash `--` used for ranges and names such as `Lewis--Skyrms`.

### dash_audit.py synopsis

```bash
python scripts/dash_audit.py manuscript/main_v2.tex
python scripts/dash_audit.py manuscript/main_v2.tex "manuscript/reviewers/*.md" --count-only
```

### dash_audit.py options

| Flag | Required | Meaning |
|:---|:---|:---|
| `files` | yes | Positional — one or more file paths or globs. |
| `--exclude-pattern REGEX`, `-x REGEX` | no | Skip occurrences whose surrounding context matches the regex. Repeatable. |
| `--context N` | no | Characters of surrounding context per occurrence (default 50). |
| `--count-only` | no | Emit per-file counts only; suppress the per-occurrence listing. |
| `--max N` | no | Exit `1` when the total dash count exceeds N. |
| `--no-skip-comments` | no | Do not strip `%` comments from `.tex` files before scanning (default: strip). |
| `--json` | no | Emit JSON instead of the text report. |
| `--soft` | no | Force exit 0 even when `--max` is exceeded. |

Exit `1` only when `--max N` is set and the total exceeds N; `2` when no input file matched; otherwise `0` (the audit is informational by default).

## bib_unused.py

Reports drift between a `.bib` file and the `\cite` calls in the manuscript.

- **Check A** — bib entries defined but never cited anywhere in the `.tex` files.
- **Check B** (with `--reverse`) — cite keys used in the `.tex` that have no entry in the `.bib`: typos or missing citations.

### bib_unused.py synopsis

```bash
python scripts/bib_unused.py manuscript/References.bib manuscript/main_v2.tex manuscript/Appendix.tex
python scripts/bib_unused.py manuscript/References.bib manuscript/main_v2.tex --reverse --strict
```

### bib_unused.py options

| Flag | Required | Meaning |
|:---|:---|:---|
| `bib_file` | yes | Positional — the `.bib` file. |
| `tex_files` | yes | Positional — one or more `.tex` files to scan for `\cite` calls. |
| `--reverse` | no | Also list cite keys used in the `.tex` with no entry in the `.bib`. |
| `--strict` | no | Exit `1` if either direction shows drift (default: only unused-bib-entry drift fails). |
| `--json` | no | Emit JSON instead of the text report. |
| `--soft` | no | Force exit 0 even on drift. |

Bib keys are matched case-sensitively and are never normalized — heterogeneous key styles (`PascalCase`, `Author_Year`, `lowercase_keyword`) are preserved as written. Exit `1` when unused entries exist (or, with `--strict`, when either direction shows drift); `2` if a file is not found; otherwise `0`.

## Recommended invocation order

Mapped onto the phases of the revise-and-resubmit workflow.

| Phase | Scripts to run | Expectation |
|:---|:---|:---|
| Checklist extracted from the narrative | `response_align.py` | Every narrative entry maps to a checklist item and vice versa. |
| After each revision round | `response_align.py`; `word_count.py --target N` and `dash_audit.py` on demand | No drift introduced by the round. |
| Final pre-submission sanity check | `response_align.py`; `word_count.py --target N`; `bib_unused.py --reverse --strict` | All exit `0` (or the word-count overshoot is documented in the cover letter). |

## Tests

Smoke tests live at `tests/test_revision_tooling.py` — 17 cases, one positive (clean input → exit 0) and one negative (deliberate drift → exit 1) per script, plus `--soft` and `--json` coverage. Run with:

```bash
pytest tests/test_revision_tooling.py -v
```

## Related documents

- `scripts/README.md` — shorter quick-start for the same four scripts.
- `templates/revision/` — drop-in skeletons for the three revision artifacts the scripts audit (operational checklist, formal response narrative, project-local LaTeX-conventions reference), plus a README describing the order of operations.
- `content/workflows/REVISION_WORKFLOW.md` — the multi-session revise-and-resubmit workflow these scripts support.
