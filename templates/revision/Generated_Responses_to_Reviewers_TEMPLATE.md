<!--
TEMPLATE-INSTRUCTIONS

This is the formal response document — the prose sent back to reviewers
alongside the resubmitted manuscript. Pair it with
`Reviewers_Responses_Checklist_TEMPLATE.md` (operational tracker) and
`paper_TEX_REF_TEMPLATE.md` (project-local LaTeX conventions). Recommended
order of operations:

  1. Copy this file into `manuscript/reviewers/` (or your project's equivalent)
     as `Generated Responses to Reviewers.md`.
  2. Fill placeholders with reviewer-text + your draft framings (Phase 1 of
     REVISION_WORKFLOW).
  3. As items close in Phase 3, expand each "Concretely, …" paragraph with
     section references and verbatim before/after quotations of the changed
     passages — those quotations are the signal that
     `scripts/response_align.py` validates against the manuscript source.
  4. Keep the per-reviewer / per-comment markers EXACT — the script parses
     `===== Reviewer #<n> =====` and `--- Comment <n>: <title> ---` to align
     entries with the operational checklist. Markdown-escaped variants
     (`\===== Reviewer \#<n> \=====`, `\--- Comment <n>: <title> \---`) are
     tolerated.

Structure preserved by this template:
  - Title + manuscript ID line
  - Per-reviewer section markers (===== Reviewer #<n> =====)
  - "--- Overall comment ---" subsection per reviewer
  - "--- Comment <n>: <title> ---" subsection per comment
  - Reviewer / Response / Concretely (before/after) three-paragraph pattern
  - Italicized verbatim quotations as *"..."* on both pre-revision (deleted)
    and post-revision (added) wording.  response_align.py auto-detects which
    is which via context cues such as "previous … has been replaced with".
  - "===== Paper Changes Checklist =====" mirror at the bottom
-->

Response to Reviewers
Manuscript [TBD: ID, e.g. JOURNAL-NNNNN]: [TBD: paper title]

\===== Reviewer \#1 \=====

\--- Overall comment \---

Reviewer:
[TBD: paste the reviewer's overall framing of the report verbatim].

Response:
[TBD: thank the reviewer and signpost the structure of what follows. Distinguish what is adopted as a revision, what is defended (with reasoning), and what is deferred].

\--- Comment 1: [TBD: short title] \---

Reviewer:
[TBD: paste the reviewer's full comment verbatim].

Response:
[TBD: narrative reply — what you accept, what you push back on (with reasoning), what is deferred].

Concretely, [TBD: section references for each touched location, plus before/after quotations for the key phrasings changed. Format: the previous wording — *"OLD"* — has been replaced at [main.tex:NN] with *"NEW"*. For structural changes (diagrams, paragraph promotions, citation additions), describe the change rather than quoting verbatim.].

\--- Comment 2: [TBD: short title] \---

Reviewer:
[TBD].

Response:
[TBD].

Concretely, [TBD].

\===== Reviewer \#2 \=====

\--- Overall comment \---

Reviewer:
[TBD].

Response:
[TBD].

\--- Comment 1: [TBD: short title] \---

Reviewer:
[TBD].

Response:
[TBD].

Concretely, [TBD].

\===== Paper Changes Checklist \=====

[TBD: mirror the operational checklist's structure for easy cross-reference. One bullet per `R<n>·C<m>` with its operational sub-actions. Keep in sync with `Reviewers Responses Checklist.md`. Example pattern:]

\[R1 · C1\] [TBD: short title]
• [TBD: atomic sub-action 1]
• [TBD: atomic sub-action 2]

\[R1 · C2\] [TBD: short title]
• [TBD: atomic sub-action].

\[R2 · C1\] [TBD: short title]
• [TBD: atomic sub-action].
