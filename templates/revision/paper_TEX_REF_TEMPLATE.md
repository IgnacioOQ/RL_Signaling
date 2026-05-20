<!--
TEMPLATE-INSTRUCTIONS

This is the project-local LaTeX-conventions reference. Its purpose is to
record the paper's specific conventions — section structure, citation style,
figure paths, notation drift across revisions — so future LLM/agent sessions
pick up cleanly without drifting toward KB-canonical idioms that don't match
the actual source.

Place at the repo root (NOT inside `manuscript/`), since the working
directory of most LLM sessions is the repo root. Conventional name:
`LP_TEX_REF.md` (Latex Paper TeX Reference); rename freely.

Recommended order of operations:
  1. Copy this file to `<repo_root>/LP_TEX_REF.md` (or your preferred name).
  2. Populate `File layout`, `Section structure`, and `Authoring conventions`
     by reading the .tex source end-to-end (Phase 0.3 of REVISION_WORKFLOW).
  3. Refresh the "Section structure" line-number table whenever the body
     shifts substantially (the table drifts with every edit). Recipe:
       grep -nE "^\\\\section\{|^\\\\subsection\{|^\\\\subsubsection\{" manuscript/main.tex
  4. Add a "Drift handling" subsection per cross-cutting symbol or
     terminology change introduced by a revision round.
  5. Use this file as a load-bearing context for every LLM edit session —
     the conventions here override KB defaults.

This file is plain markdown — no YAML frontmatter (it's a project reference,
not a KB-managed plan).
-->

# [TBD: paper short-name] — LaTeX Conventions Reference

Project-local LaTeX-conventions reference for [TBD: manuscript ID / title].
Load this file at the start of every LLM session that edits `.tex` source.

## File layout

- `manuscript/main.tex` — [TBD: working revision filename — `main_v2.tex` if you use the working-copy convention].
- `manuscript/Appendix.tex` — [TBD: appendix file, if any; note whether it compiles standalone].
- `manuscript/References.bib` — [TBD: bibliography file, sources shared between main and appendix].
- `manuscript/reviewers/` — [TBD: directory holding the operational checklist and formal response narrative].
- `[TBD: other dirs, e.g. analytics/, figures/, results/]`.

## Section structure

Refresh with:

```bash
grep -nE "^\\\\section\{|^\\\\subsection\{|^\\\\subsubsection\{" manuscript/main.tex
```

| Line | Level | Heading |
|---:|:---|:---|
| NNN | section | [TBD: title] |
| NNN | subsection | [TBD: title] |

[TBD: keep a sentence per non-obvious section explaining its argumentative role.]

## Authoring conventions

### Citations and bibliography

- Citation commands in use: [TBD: `\citep{}`, `\citet{}`, `\cite{}` — note which].
- Bibkey style is heterogeneous: [TBD: examples and rules. E.g. `PascalCase`, `Author_Year`, `lowercase_keyword`. Tools must NOT normalize keys].
- Always re-use existing keys rather than renaming. If a paper appears in both styles, pick one and unify.

### Figures

- `\graphicspath{}` setting: [TBD].
- Figure paths convention: [TBD, e.g. `results/<short_name>.png`].
- Captions: [TBD: project-specific conventions].

### Math notation

- [TBD: list of project-specific symbol conventions, e.g. `u_i` for per-agent payoffs, `Sig` for signal alphabet].
- [TBD: argument-order conventions, e.g. `u_i(x, y, a_i)` — states first, own action last].

### Line discipline

- [TBD: paragraph-per-line vs sentence-per-line in the source]. Edits should preserve the existing pattern.

### Em-dash policy

- [TBD: `---` (LaTeX em-dash) for genuinely abrupt breaks or strong parentheticals; commas / parentheses for routine parentheticals. Run `scripts/dash_audit.py manuscript/main.tex` to audit].

### Quotation marks

- [TBD: LaTeX `` ``...'' `` for double quotes, `` `...' `` for single. Avoid straight `"`/`'` in `.tex` source].

## Drift handling

[TBD: per-revision-round subsections recording notation or terminology changes that should propagate. Example pattern:]

### Drift introduced by Reviewer N's comment M

- [TBD: change, e.g. `G_i` → `u_i` for per-agent payoffs. Apply globally in §1.3 and Diagram 2; verify §3 / §4 if either references the old form].

## Compile and display

```bash
cd manuscript/
latexmk -pdf -interaction=nonstopmode main.tex
```

- Expected page count: [TBD].
- Known harmless warnings: [TBD: e.g. "Hfootnote.N has been referenced but does not exist" is hyperref cold-start noise].

## Build artifacts

Recommended `.gitignore` entries (paste into the repo's `.gitignore` as needed):

```gitignore
# LaTeX build artifacts
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.synctex.gz
```

The `.tex`, `.bib`, and final `.pdf` are typically committed; the rest are reproducible from `latexmk`.

## Gotchas

[TBD: paper-specific traps that surfaced during the revision and would burn the next agent's time. Examples: peculiar package interactions, figure-render conventions, footnote-numbering oddities, citation-style quirks. Empty until something surfaces.]
