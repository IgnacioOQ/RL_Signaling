# `templates/revision/` — paper-revision starter kit

A drop-in skeleton set for a new "revise and resubmit" round. Copying these
files into a fresh paper's `manuscript/reviewers/` directory (plus
`paper_TEX_REF_TEMPLATE.md` at the repo root) gives a structurally complete
starting point that can be filled in.

Designed to pair with the cross-paper revision-tooling scripts at
[`../../scripts/`](../../scripts/) — see [`../../scripts/README.md`](../../scripts/README.md).

## Files in this directory

| Template | Copy to | Purpose |
|---|---|---|
| `Reviewers_Responses_Checklist_TEMPLATE.md` | `manuscript/reviewers/Reviewers Responses Checklist.md` | Operational `.tex`-tracking checklist (atomic edits, top-line progress, sub-action checkboxes). |
| `Generated_Responses_to_Reviewers_TEMPLATE.md` | `manuscript/reviewers/Generated Responses to Reviewers.md` | Formal response narrative sent to reviewers (per-comment Reviewer/Response/Concretely structure). |
| `paper_TEX_REF_TEMPLATE.md` | `<repo_root>/LP_TEX_REF.md` (or similar) | Project-local LaTeX conventions reference — section structure, citation style, em-dash policy, notation drift across rounds. |

## Order of operations

1. **Copy the templates** into the locations above.
2. **Fill the YAML frontmatter** in the operational checklist (description,
   repository, owner, last_checked).
3. **Write the formal narrative first** (Phase 1 of REVISION_WORKFLOW.md):
   reviewer comment + accept/defend/defer response. Iterate with the user
   until the framing is right.
4. **Extract the operational checklist** from the narrative (Phase 2): one
   `### [R<n> · C<m>]` block per comment, with sub-action checkboxes.
5. **Run the Phase 3 sub-loop** per item: pick item → propose 2–3 edit
   options → user picks → apply → compile → tick `[x]` + update the
   "Concretely, ..." paragraph in the narrative with before/after quotations.
6. **Use the audit scripts throughout**:
   - `scripts/response_align.py` after every round → catches drift between
     the three coupled documents.
   - `scripts/word_count.py --target N` on demand → tracks the body word
     count against the journal limit.
   - `scripts/dash_audit.py` on demand → tracks em-dash overuse.
   - `scripts/bib_unused.py --strict` as a pre-submission check → trims
     never-cited entries and catches typos in cite keys.

## Structural conventions preserved by the templates

Recorded here so future agents can re-derive them without reading every
.md file end-to-end. These match what `scripts/response_align.py` parses.

### Operational checklist
- Per-comment heading format: `### [R<n> · C<m>] <title>` (regular spaces
  around `·`; non-breaking space variants tolerated).
- Top-line progress table — one row per comment, status cell
  `` `[x]`/`[~]`/`[ ]` ``.
- Sub-bullet format: `- [x] <description>. — *<notes>* (done YYYY-MM-DD)`.
- Manuscript anchors as markdown links: `[main.tex:NNN](../main.tex#L<NNN>)`.

### Formal response narrative
- Per-reviewer section markers: `===== Reviewer #<n> =====` (or escaped
  `\===== Reviewer \#<n> \=====`).
- Per-comment subsection markers: `--- Comment <n>: <title> ---` (or
  escaped `\--- Comment <n>: <title> \---`).
- Verbatim manuscript quotations as `*"..."*` (italic + double-quote).
- Pre-revision (deleted) quotes appear in "the previous wording — *X* —
  has been replaced with *Y*" patterns; `response_align.py` auto-detects
  these via context cues and skips them in the verbatim-quote drift check.

### LaTeX conventions reference (`LP_TEX_REF.md`)
- Plain markdown, no YAML frontmatter.
- Section order: File layout → Section structure → Authoring conventions
  (citations, figures, math, line discipline) → Drift handling → Compile
  and display → Build artifacts → Gotchas.
- Section-structure line numbers drift; refresh with:

  ```bash
  grep -nE "^\\\\section\{|^\\\\subsection\{|^\\\\subsubsection\{" manuscript/main.tex
  ```

## Related KB docs

If you have access to a kb_mcp instance:

- `content/workflows/REVISION_WORKFLOW.md` — the full multi-session workflow.
- `content/how-to/LATEX_WRITING_SKILL.md` — KB-canonical LaTeX conventions
  (deviate freely; record deviations in `LP_TEX_REF.md`).
