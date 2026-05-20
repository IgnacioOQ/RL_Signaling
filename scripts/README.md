# `scripts/` — cross-paper revision tooling

Four standalone Python 3.10+ scripts that audit a paper-revision arc.
Distinct from `analytics/scripts/`, which holds this paper's analytics —
these scripts operate on the manuscript / response / checklist orbit and
are designed to copy verbatim into another paper's repository.

Paired with the templates at [`../templates/revision/`](../templates/revision/).
Built per [TODO_WORKFLOW.md](../TODO_WORKFLOW.md)'s `todo.revision_tooling` task.

## Quick reference

| Script | Purpose | Triggered by |
|---|---|---|
| `response_align.py` | Drift checker across the three coupled artifacts (operational checklist, formal response narrative, manuscript). | Run after every Phase 3 round of REVISION_WORKFLOW. |
| `word_count.py` | `texcount` wrapper with target comparison and per-section breakdown. | On demand during `todo.word_count_reduction`. |
| `dash_audit.py` | Em-dash counter (LaTeX `---` and Unicode `—`; ignores en-dashes `--`). | On demand during `todo.dash_sweep` and voice-revision passes. |
| `bib_unused.py` | Defined-but-never-cited bib entries (and, with `--reverse`, cited-but-not-defined keys). | Pre-submission bibliography hygiene. |

## Exit-code policy

By default each script exits **non-zero on drift** so it can be wired into
CI or pre-commit later:

- `response_align.py` → exit 1 if any of the four drift sections has hits.
- `word_count.py` → exit 1 if body word count exceeds `--target`.
- `dash_audit.py` → exit 1 only if `--max N` is set and the total exceeds N
  (audit is otherwise informational).
- `bib_unused.py` → exit 1 if any bib entry is defined but never cited;
  with `--strict`, also fail on cited-but-undefined keys.

Pass `--soft` to any script to force exit-0 (useful when you want the
report but don't want a failing exit in a script-runner).

## Common flags

All four scripts share:

- `--help` — full usage.
- `--json` — machine-readable output for diffing across revisions or
  wiring into CI.
- `--soft` — never exit non-zero (defaults to exit-1 on drift).

## CLI samples

```bash
# Drift check — run after every Phase 3 round.
python scripts/response_align.py \
  --checklist  "manuscript/reviewers/responses_checklist.md" \
  --responses  "manuscript/reviewers/responses_to_reviewers.md" \
  --manuscript manuscript/main_v2.tex manuscript/Appendix.tex

# Strict verbatim-quote check (includes pre-revision quotes by default skipped).
python scripts/response_align.py [...same as above...] --strict-quotes

# Word count vs journal target.
python scripts/word_count.py manuscript/main_v2.tex --target 9000 --per-section

# Em-dash audit.
python scripts/dash_audit.py manuscript/main_v2.tex
python scripts/dash_audit.py manuscript/main_v2.tex "manuscript/reviewers/*.md"
python scripts/dash_audit.py manuscript/main_v2.tex --exclude-pattern '\\citep\{[^}]*--[^}]*\}'

# Bib hygiene.
python scripts/bib_unused.py manuscript/References.bib manuscript/main_v2.tex manuscript/Appendix.tex
python scripts/bib_unused.py manuscript/References.bib manuscript/main_v2.tex --reverse --strict
```

## Recommended invocation order during a revision

1. **Phase 0 (setup)**: none.
2. **Phase 1 (response narrative drafted)**: none yet — the operational
   checklist doesn't exist.
3. **Phase 2 (checklist extracted)**: `response_align.py` once, to confirm
   every narrative entry maps to a checklist item and vice versa.
4. **Phase 3 (iterative)**: after each round of edits, `response_align.py`
   to catch drift introduced by the round. `word_count.py --target N` and
   `dash_audit.py` on demand.
5. **Phase 4 (final sanity check)**:
   - `response_align.py` — exit 0 expected.
   - `word_count.py --target N` — exit 0 expected (or document the
     overshoot in the cover letter).
   - `bib_unused.py --strict` — exit 0 expected; if there are unused
     entries, decide whether to trim them.
6. **Phase 5 (knowledge capture)**: not script-driven.

## Design constraints

- **Python 3.10+** floor.  No third-party dependencies beyond `pytest`
  (used by `tests/test_revision_tooling.py`).
- **No coupling** to this paper.  File paths are CLI arguments, not
  hardcoded. Copy the four `.py` files into another paper's `scripts/` —
  they should work as-is.
- **Standalone**: no shared internal module between the scripts.

## Tests

```bash
pytest tests/test_revision_tooling.py -v
```

Smoke tests per script cover one positive case (clean input, expected
exit-0) and one negative case (deliberate drift, expected exit-1).
