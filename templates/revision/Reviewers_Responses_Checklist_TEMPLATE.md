<!--
TEMPLATE-INSTRUCTIONS

This is the operational checklist that tracks atomic edits to the manuscript
source. Pair it with `Generated_Responses_to_Reviewers_TEMPLATE.md` and
`paper_TEX_REF_TEMPLATE.md`. Recommended order of operations:

  1. Copy this file into `manuscript/reviewers/` (or your project's equivalent)
     as `Reviewers Responses Checklist.md`.
  2. Fill the YAML frontmatter (description, repository).
  3. Replace `[TBD: ...]` placeholders with project content.
  4. Add one `### [R<n> · C<m>]` block per reviewer comment.
  5. As you tick `[x]` items in Phase 3, run `scripts/response_align.py` to
     verify the operational checklist, the formal response narrative, and the
     manuscript stay in sync.

Structure preserved by this template:
  - YAML frontmatter (plan schema)
  - Status legend with [x] / [~] / [ ] flavors
  - Top-line progress table (one row per comment)
  - "§<n>.<m>-dependent response entries" ledger (verbatim / paraphrase / pointer)
  - Per-comment sections (### [R<n> · C<m>])
  - "Notation drift introduced during the revision" section
  - "Build / verification" section
-->

---
status: active
type: plan
id: [TBD: short-id, e.g. paper_xyz.revision_checklist]
description: [TBD: one-sentence description of what this checklist tracks]
label: [planning, agent]
injection: excluded
volatility: evolving
scope: project-specific
owner: agent
last_checked: 'YYYY-MM-DD'
---

# Reviewers Responses Checklist — [TBD: Manuscript ID]

Operational tracker for the revision pass on [main.tex](../main.tex). Companion to:

- [Generated Responses to Reviewers.md](Generated%20Responses%20to%20Reviewers.md) — formal response document (the prose sent to reviewers).
- [TBD: prior-submission response doc, if any].

Every reviewer comment is listed below with its sub-actions. Items already landed in `main.tex` are marked `[x]` with the date; items still open are `[ ]`; items partially landed are `[~]` with a note on what remains.

## Status legend

- `[x]` — fully landed in `main.tex` (date in parentheses)
- `[~]` — partially landed (remaining work noted inline)
- `[ ]` — open

## Top-line progress

| Comment | Title | Status |
|---|---|---|
| R1·C1 | [TBD: short title] | `[ ]` |
| R1·C2 | [TBD: short title] | `[ ]` |
| R2·C1 | [TBD: short title] | `[ ]` |

## §[TBD: section]-dependent response entries — re-verify if §[TBD] is rewritten

[TBD: list response-narrative entries that quote or reference this section. Strength legend:]
**verbatim** = quotation must be updated in lockstep with the section;
**paraphrase** = re-verify framing still applies;
**pointer** = survives any prose change short of removing the section.

- **R<n>·C<m>** — **[verbatim | paraphrase | pointer]**: [TBD: brief note on which response entry depends on which content].

---

## Reviewer 1

Overall framing: [TBD: one-sentence characterization of this reviewer's report].

### [R1 · C1] [TBD: short title]

**Reviewer's concern:** [TBD: one-paragraph pull-quote summarizing the comment].

**Actions:**

- [ ] [TBD: atomic edit, with target location e.g. §1.1 or `[main.tex:NN]`]. — *[TBD: notes; when ticking `[x]`, add a brief description of what landed]*.
- [ ] [TBD: another atomic edit].

### [R1 · C2] [TBD: short title]

**Reviewer's concern:** [TBD].

**Actions:**

- [ ] [TBD].

---

## Reviewer 2

Overall framing: [TBD].

### [R2 · C1] [TBD: short title]

**Reviewer's concern:** [TBD].

**Actions:**

- [ ] [TBD].

---

## Notation drift introduced during the revision

[TBD: cross-cutting symbol or terminology changes that the revision introduces. Examples: `G_i` → `u_i`; `X, Y` reframed as state alphabets rather than random variables; new convention for argument order. Empty until the revision starts introducing drift; populated as decisions land. Mirror durable drift into `LP_TEX_REF.md`.]

## Build / verification

After each round of edits, recompile and confirm clean output:

```bash
cd manuscript/
latexmk -pdf -interaction=nonstopmode main.tex
```

[TBD: project-specific build notes, e.g. expected page count, known harmless warnings.]
