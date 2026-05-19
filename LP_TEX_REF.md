# Local Project LaTeX Conventions (LP_TEX_REF)
- status: active
- type: reference
- id: rl_signaling.lp_tex_ref
- description: Local LaTeX conventions used by manuscript/main.tex (PHOS-17993, "Signaling Games with Distributed Rewards"). Authoring rules, citation patterns, label prefixes, cross-reference style, sectioning, and known gotchas. Companion to (and deliberate deviation from) content/how-to/LATEX_WRITING_SKILL.md in the KB.
- label: [reference, project]
- injection: directive
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-18
<!-- content -->

This file is the project-local source of truth for how the LaTeX paper at [manuscript/main.tex](manuscript/main.tex) is written. The KB's [content/how-to/LATEX_WRITING_SKILL.md](https://example/LATEX_WRITING_SKILL.md) prescribes more disciplined conventions (sentence-per-line, `\cref{}`, `authorlastname_year_keyword`); those are deliberately **not** adopted here, because the existing paper has its own established style and PHOS-17993 is a one-shot philosophy submission, not a long-lived multi-paper repo. The matching call lives in the memory file [`feedback_paper_work.md`](.claude/projects/.../memory/feedback_paper_work.md) Rule 8. If you find yourself wanting to refactor `main.tex` to "match the KB", stop — re-read Rule 8 first.

## File layout

- **`manuscript/main.tex`** — the canonical paper source. ~640 lines, 12pt article, single-file body (no `\input{}` includes used yet).
- **`manuscript/main_v2.tex`** — a working copy of `main.tex` for in-progress edits. The "v2" is a versioning marker, not a fork; once a `_v2` revision is accepted, the older file is deleted and `_v2` renames to the canonical name (see the 2026-05-18 V1 → V2 transition of `Proof of Concept (Paper Draft).md` for the precedent).
- **`manuscript/section_2_3.tex`** — *(deleted 2026-05-19)*. Was a standalone fragment of §2.3 drafted as a drop-in `\input{}`. Its content was ported into `main_v2.tex` on 2026-05-18, the fragment was retained as a reference copy through 2026-05-19, then deleted when the user confirmed `main_v2.tex` is the single source of truth for §2.3. The "Known drift in `section_2_3.tex`" section below documents what the drift was; it is historical now.
- **`manuscript/References.bib`** — the BibTeX file wired up by `\bibliography{References}` (line 643 of `main.tex`). 257 entries (a large personal / cross-paper library); only ~28 of them are actually `\cite`d from `main.tex`. Before adding a new `\cite` key, grep this file first — odds are good the author you want is already in there under some existing key, possibly from an unrelated paper.

## Section structure of `main.tex`

| Line | Header | Notes |
|---|---|---|
| 101 | `\section{Introduction: Signal-Trading Games}` | §1 |
| 103 | `\subsection{Signaling Games}` | §1.1 |
| 177 | `\subsection{Goal-Sharing}` | §1.2 |
| 188 | `\subsection{Signal-Trading Games}` | §1.3 — establishes the agent / signal / action / game tuple notation that §2.3 must inherit |
| 280 | `\section{Formal Setup and Proof of Concept}` | §2 |
| 282 | `\subsection{Informal Proof of Concept}` | §2.1 |
| 291 | `\subsection{Formal Setup}` | §2.2 |
| 293 | `\subsubsection{The Problem Space}` | §2.2.1 |
| 322 | `\subsubsection{The Agents}` | §2.2.2 |
| 429 | `\subsubsection{(Normalized) Mutual Information}` | §2.2.3 |
| 476 | `\subsection{Proof of Concept}` | **§2.3 — the active rewrite target** |
| 505 | `\section{Simulation Results}` | §3 |
| 525 | `\subsection{Simulations with Matching Games}` | §3.1 |
| 568 | `\subsection{Simulations with Random Games}` | §3.2 |
| 618 | `\section{Discussion and Conclusion}` | §4 |
| 642–643 | `\bibliographystyle{plainnat}` / `\bibliography{References}` | bibliography lives in `References.bib` (not in repo) |

All subsections and subsubsections are **numbered** (no starred `\section*` / `\subsection*` / `\subsubsection*`). Match this style when adding new internal headers under §2.3.

## Authoring conventions

### Line discipline

- **One paragraph per line.** A `<p>` of body prose lives on a single long line, not wrapped at column width and not broken at sentence boundaries. See `main.tex` lines 274, 499, 527, 612 for canonical examples.
- This is intentional and is the opposite of the KB skill's "one sentence per line." Reason: the existing paper was authored this way, and rewrapping every paragraph would generate ~600 lines of diff noise for zero rendered-output gain.
- **Implication for editing**: when changing a single sentence inside a long paragraph line, you must edit the whole line. Use `Edit` with a long `old_string` that uniquely locates the sentence in its surrounding context.

### Sectioning headers

- `\section{...}`, `\subsection{...}`, `\subsubsection{...}` — all numbered. No starred variants used anywhere in `main.tex`.
- No `\label{sec:...}` exists on any section header in `main.tex`. Sections are cross-referenced in prose as literal Unicode "§1.3" / "§2.3" / etc., not via `\ref{}`. Match this — don't introduce `\label{sec:...}` and `\ref{sec:...}` pairs unless you also add labels to every existing section (out of scope).

### Cross-references

| Object | Pattern | Example from `main.tex` |
|---|---|---|
| Figure | `Figure \ref{fig:<key>}` | `Figure \ref{fig:canonical_figures_1x2}` (line 499) |
| Table / Game | `Table \ref{game:<key>}` | `Table \ref{game:strangers_game}` (line 274) |
| Section | literal `§X.Y` in prose | "see §3", "§1.3 introduces…" |

- **Always `\ref{}`**, never `\autoref{}` or `\cref{}`. The `hyperref` package is loaded (line 32) with `colorlinks=true` so the `\ref` already produces a clickable colored link. `cleveref` is **not** loaded; don't introduce it.
- **Always prefix with the literal noun in prose** ("Figure ", "Table ", with a space, then `\ref{}`). The KB skill would have you use `\cref{}` to get this automatically; the local convention writes the noun out.

### Label prefixes

Observed in `main.tex`:

- `fig:` — figures (e.g. `fig:canonical_figures_1x2`, `fig:example_run`, `fig:canonical_figures`, `fig:random_figures`).
- `game:` — game tables / Strangers' Game style (e.g. `game:strangers_game`).
- No `sec:`, `subsec:`, `tab:`, `eq:`, `alg:` labels exist.

When introducing a new figure label, follow the `fig:<snake_case_descriptor>` pattern. If you need a different kind of label (e.g. an equation), pick a prefix consistent with the kind, but flag the new prefix in this doc on first use.

## Citations and bibliography

### Package + style

`main.tex` preamble (line 71):

```latex
\usepackage[numbers]{natbib}  % or [authoryear], adjust as needed
```

`main.tex` bibliography block (lines 642–643):

```latex
\bibliographystyle{plainnat}  % or abbrvnat, unsrtnat, etc.
\bibliography{References}
```

**Effect**: every citation renders as `[N]` (numeric, square brackets), regardless of whether you write `\cite{key}` or `\citep{key}`. The `[numbers]` natbib option makes `\citet` and `\citep` produce identical output to `\cite`. So:

- `\cite{Argiento2009}` → `[1]`
- `\citep{Argiento2009}` → `[1]`
- `\citet{Argiento2009}` → `Argiento et al. [1]` (only `\citet` differs under numeric mode, by adding the author name)

In practice `main.tex` uses `\cite` and `\citep` interchangeably; pick whichever already appears in the surrounding context. Author names are spelled out in prose ("I am here following Huttegger's argument \cite{Huttegger_2007}") rather than relying on the citation command to produce them.

### Bib key conventions

There is **no convention.** Existing keys are heterogeneous and date back across multiple sessions of paper writing. Examples observed in `main.tex`:

| Style | Examples |
|---|---|
| `Authorlastname` + `Year` (PascalCase, no separator) | `Argiento2009`, `Catteeuw2013`, `Lewis1969`, `RothErev1995`, `Skyrms2010`, `SuttonBarto1998` |
| `Authorlastname_Year` (underscore) | `Huttegger_2007` |
| `authorlastnameYYYYkeyword` (lowercase) | `grice1975logic`, `gilbert1990walking`, `gilbert2008social`, `crawford1982strategic`, `huttegger2014probe`, `kane2015handicap`, `zollman2012rspb`, `millikan2021neuroscience`, `lacroix2025information`, `steinert2016compositional`, `noukhovitch2021emergentcommunicationcompetition`, `head_scikit_optimize_2021`, `skyrms1996evolution`, `skyrms2010signals`, `skyrms_signals` |
| ALLCAPS keys | `BLUME1993547`, `FreebornForthcoming-FRECUI-2`, `HerrmannForthcoming-HERSTS-2`, `Simons2019-SIMNCA-4`, `Taylor1978EvolutionarilySS` |
| Underscored descriptor keys | `zollman_signaling`, `ucb_regret` |

**How to apply**:

1. Before introducing a new `\cite{NEW_KEY}`, grep `main.tex` to see whether the author/work is already cited under some existing key. Reuse the existing key verbatim.
2. If the work is genuinely new, ask the user to confirm the bib key they want — do not invent one to the KB's `authorlastname_year_keyword` template, because that would clash with the heterogeneous existing keys.
3. **Capitalization is sometimes load-bearing.** `Argiento2009` (capital A) is the existing key; `argiento2009` (lowercase) would be a different, undefined key and would produce a `?` in the rendered PDF.

### Known bib keys already cited in `main.tex`

Full inventory (for grep / reuse):

```
Argiento2009, BLUME1993547, Catteeuw2013, FreebornForthcoming-FRECUI-2,
HerrmannForthcoming-HERSTS-2, Huttegger_2007, Lewis1969, RothErev1995,
Simons2019-SIMNCA-4, Skyrms2010, SuttonBarto1998, Taylor1978EvolutionarilySS,
crawford1982strategic, gilbert1990walking, gilbert2008social, grice1975logic,
head_scikit_optimize_2021, huttegger2014probe, kane2015handicap,
lacroix2025information, millikan2021neuroscience,
noukhovitch2021emergentcommunicationcompetition, skyrms1996evolution,
skyrms2010signals, skyrms_signals, steinert2016compositional, ucb_regret,
zollman2012rspb, zollman_signaling
```

Pemantle is **not** currently cited in `main.tex`, and `References.bib` has no standalone `Pemantle*` entry — Robin Pemantle appears only as a co-author of `Argiento2009`. The §2.3 draft's `\citep{pemantle2007}` would therefore render as `[?]` (undefined). Two options when porting §2.3:

1. **Drop the Pemantle citation.** The sentence "follows from Pemantle's stable-manifold theorem" can stand without a separate citation, since the theorem is already attributed in prose and the Argiento paper (already cited at `Argiento2009`) is the upstream application. This is the lower-friction option.
2. **Add a new bib entry.** If you want the stable-manifold theorem cited directly, add an entry for Pemantle's 2007 *Probability Surveys* paper ("A survey of random processes with reinforcement") to `References.bib`. Pick a key that fits the existing heterogeneous style — `Pemantle2007` (PascalCase, matches `Argiento2009`, `Skyrms2010`) is the most consistent choice.

## Figures

### Graphics search path

`main_v2.tex` declares `\graphicspath{}` right after `\usepackage{graphicx,...}` (just past line 24 of the preamble):

```latex
\graphicspath{%
  {../results/legacy/plots/}%
  {../results/proof_of_concept/}%
  {../results/new_code/plots/}%
}
```

This lets every `\includegraphics{}` in the body use a **bare filename** — pdflatex looks the file up in each path in order, taking the first hit. Paths are relative to the `.tex` file's directory (`manuscript/`).

**Implication**: figures with name collisions across the three directories would resolve to `legacy/plots/` first. As of 2026-05-18 no collisions exist. If a future figure name collides, either rename one of the files or add a more specific path prefix in `\includegraphics{path/file.png}`.

### Figure name mapping

`main.tex` was originally written when figures lived under short bare names alongside the `.tex`. The 2026-05-15 `results/` reorg renamed the canonical / complex-randomized panels to long descriptive names under `results/legacy/plots/`. On 2026-05-18 each `\includegraphics{old.png}` in `main_v2.tex` was updated to point at the actual current file (the active 3×2 grids around lines 547–559 and 591–603, and both commented-backup sibling blocks updated in parallel so the backups stay in sync). The mapping for reference:

| Old bare name in `main.tex` | Actual file (in `results/legacy/plots/`) |
|---|---|
| `init_smooth_r.png` | `init_smooth_r.png` (added 2026-05-18) |
| `init_smooth_nmi.png` | `init_smooth_nmi.png` (added 2026-05-18) |
| `example_process_rewards.png` | `example_process_rewards.png` (added 2026-05-18) |
| `example_process_nmi.png` | `example_process_nmi.png` (added 2026-05-18) |
| `urn_canon_reward.png` | `Roth-Erev_canonical_Agent_0_final_reward.png` |
| `urn_canon_nmi.png` | `Roth-Erev_canonical_Agent_0_NMI.png` |
| `urn_canon_regression.png` | `Roth-Erev_canonical_regression_signals_True_fullinfo_False.png` |
| `q_canonical_reward.png` | `Q-learning_canonical_Agent_0_final_reward.png` |
| `q_canon_nmi.png` | `Q-learning_canonical_Agent_0_NMI.png` |
| `q_canon_regression.png` | `Q-learning_canonical_regression_signals_True_fullinfo_False.png` |
| `urn_complex_random_reward.png` | `Roth-Erev_complex_randomized_Agent_0_final_reward.png` |
| `urn_complex_random_nmi.png` | `Roth-Erev_complex_randomized_Agent_0_NMI.png` |
| `urn_complex_random_regression.png` | `Roth-Erev_complex_randomized_regression_signals_True_fullinfo_False.png` |
| `q_complex_random_reward.png` | `Q-learning_complex_randomized_Agent_0_final_reward.png` |
| `q_complex_random_nmi.png` | `Q-learning_complex_randomized_Agent_0_NMI.png` |
| `q_complex_random_regression.png` | `Q-learning_complex_randomized_regression_signals_True_fullinfo_False.png` |

The first four kept their short bare names — user added them under those names directly. The remaining twelve were renamed during the reorg. The mapping rules that did the rename:

- `urn_*` → `Roth-Erev_*` (Roth-Erev is the urn-based learner; user renamed to be explicit about the algorithm).
- `q_*` / `q_canon_*` → `Q-learning_*`.
- `*_canon_*` / `*_canonical_*` → `*_canonical_*` (unified spelling).
- `*_complex_random_*` → `*_complex_randomized_*` (post-Bug 6, the complex experiment was made randomized; the file rename reflects that).
- `*_reward` → `*_Agent_0_final_reward` (the new producer saves per-agent final-reward histograms; "Agent_0" is the convention since both agents are symmetric).
- `*_nmi` → `*_Agent_0_NMI` (capital NMI).
- `*_regression` → `*_regression_signals_True_fullinfo_False` (the partial-info, signals-on slice — matches the prose at line 527, "signals are present and information is partial").

When new figures are added: prefer the long descriptive name in `results/<subdir>/`. The `\graphicspath{}` mechanism makes bare names work, so the figure-include line stays readable.

### §2.3 figure (ported into `main_v2.tex` on 2026-05-18)

§2.3 of `main_v2.tex` now uses the V2 prose with Option F as the sole figure. The include line is:

```latex
\includegraphics[width=\linewidth]{proof_of_concept_plot_RE.png}
```

(`\graphicspath{}` resolves the bare filename via `../results/proof_of_concept/`.) The figure label is `fig:proof-of-concept`. The four internal headings in §2.3 (*The figure*, *Three observations*, *Reading*, *What this is, and what this is not*) use `\paragraph{...}` run-in style rather than `\subsubsection{...}` to keep §2.3 visually unified — if you'd prefer the numbered hierarchy that §2.2 uses, swap all four.

`section_2_3.tex` was the original standalone fragment with its own absolute path `../results/proof_of_concept/proof_of_concept_plot_RE.png`. The port into `main_v2.tex` (2026-05-18) dropped the prefix, dropped the `\citep{pemantle2007}` (no bib entry), normalized `\citep{argiento2009}` → `\citep{Argiento2009}`, and replaced `\autoref{}` with `Figure \ref{}` per local style. The fragment itself was deleted on 2026-05-19; `main_v2.tex` is the single source of truth for §2.3 going forward.

### Appendix.tex figure mapping (2026-05-18)

`Appendix.tex` is a standalone document (its own `\documentclass` and `\end{document}`) that compiles to `Appendix.pdf` separately from `main_v2.pdf`. It has its own `\graphicspath{}` block in the preamble (right after `\usepackage{graphicx}`), pointing at the same three directories as `main_v2.tex` so figures can be referenced by bare name.

14 `\includegraphics{}` calls across three sections (Costly Signals, TD-Learning, Optimization). The naming convention is **mixed by design** — Sections A and C, plus the complex-randomized column of Section B, use **bare names** (matching files the user added directly to `results/legacy/plots/`); the canonical column of Section B uses the **long descriptive names** because no bare-named canonical TD file exists in the figure store.

| Include in `Appendix.tex` | Resolved file in `results/legacy/plots/` | Style |
|---|---|---|
| `q_rewards_costlysignal.png` | `q_rewards_costlysignal.png` | bare |
| `q_signalusage_costlysignal.png` | `q_signalusage_costlysignal.png` | bare |
| `q_rewardvscost_costlysignal.png` | `q_rewardvscost_costlysignal.png` | bare |
| `q_nmivscost_costlysignal.png` | `q_nmivscost_costlysignal.png` | bare |
| `TD-learning_canonical_Agent_0_final_reward.png` | `TD-learning_canonical_Agent_0_final_reward.png` | long |
| `td_complex_reward.png` | `td_complex_reward.png` | bare |
| `TD-learning_canonical_Agent_0_NMI.png` | `TD-learning_canonical_Agent_0_NMI.png` | long |
| `td_complex_nmi.png` | `td_complex_nmi.png` | bare |
| `TD-learning_canonical_regression_signals_True_fullinfo_False.png` | same | long |
| `td_complex_random_regression.png` | `td_complex_random_regression.png` | bare |
| `q_opt_canonical.png` | `q_opt_canonical.png` | bare |
| `td_opt_canonical.png` | `td_opt_canonical.png` | bare |
| `q_opt_games.png` | `q_opt_games.png` | bare (was `q_opt_random.png` in source; renamed to match the actual file) |
| `td_opt_games.png` | `td_opt_games.png` | bare (was `td_opt_random.png` in source; renamed to match) |

**Note on the bare-vs-long inconsistency**: §2's canonical TD column uses the long form because no `td_canonical_{reward,nmi,regression}.png` exists in `results/legacy/plots/`. If you'd like the Appendix entirely on bare names for visual consistency, add three bare-name aliases (copies or symlinks of the existing `TD-learning_canonical_*` files): `td_canonical_reward.png`, `td_canonical_nmi.png`, `td_canonical_regression.png`. Once those exist, three `\includegraphics{}` calls in `Appendix.tex` Section B can be flipped back to the bare names with a single `replace_all` Edit each.

**Unreferenced extras in `results/legacy/plots/`**: `q_nmi_costlysignal.png`, `td_complex_random_nmi.png`, `td_complex_random_reward.png`, `td_complex_regression.png` are present but not cited from `Appendix.tex`. Harmless. May be intentional variants or candidates for future figure additions.

## Math notation (matching §1.3)

The notation §2.3 must inherit from §1.3 (`main.tex` lines 188–280, especially the four-step game description in the diagram cell):

| Concept | Symbol |
|---|---|
| Agents | $A_1$, $A_2$ |
| Random variables | $X$, $Y$ (independent binary) |
| Realized observations | $x$, $y$ |
| Signal sets | $Sig_1$, $Sig_2$ |
| Signals | $s_1$, $s_2$ |
| Signaling functions | $f_1(x)$, $f_2(y)$ |
| Action sets | $Ac_1$, $Ac_2$ |
| Actions | $a_1$, $a_2$ |
| Action functions | $g_1(x, s_2)$, $g_2(y, s_1)$ |
| Games | $G_1$, $G_2$ |
| Rewards | $r_1$, $r_2$; $G_i(a_i, x, y) = r_i$ |

- **Subscripts** for agent index ($f_1$, not $f^{(1)}$).
- **Parentheses** for function application ($f_1(x)$, not $f_1[x]$).
- **1-indexed** ($1, 2$ — never $0, 1$).
- **Optimal action map**: use $a_i^\star$, not $\alpha_i^\star$.
- **World state**: the pair $(x, y)$, not a separate symbol $\mathbf{v}$.

§2.3 was rewritten on 2026-05-18 to use this notation throughout `main_v2.tex`.

## Drift handling for `section_2_3.tex` (resolved 2026-05-18, fragment deleted 2026-05-19)

`section_2_3.tex` was drafted as a standalone fragment with four KB-style drifts from local convention. All four were translated when §2.3 was ported into `main_v2.tex` on 2026-05-18:

1. **`\autoref{fig:proof-of-concept}`** → `Figure \ref{fig:proof-of-concept}` (literal noun + `\ref{}`).
2. **`\citep{argiento2009}`** → `\citep{Argiento2009}` (capital A — matches the existing key in `main.tex`).
3. **`\citep{pemantle2007}`** → dropped entirely. `References.bib` has no Pemantle entry; Pemantle's stable-manifold theorem is now attributed in prose only, with the upstream `Argiento2009` citation carrying the formal reference.
4. **`\subsubsection*{...}`** (four of them) → `\paragraph{...}` run-in style, keeping §2.3 visually unified. (`\subsubsection{...}` numbered was the alternative; flagged for swap if you want sub-numbering.)

Historical note kept here in case §2.3 is ever re-drafted from scratch. The fragment file itself was deleted on 2026-05-19; `main_v2.tex` is the single source of truth.

## Argiento footnote drop (2026-05-19)

Earlier drafts of §2.3 carried a footnote at the "What this is not" paragraph reading *"The precise step that breaks down, and the routes that might recover a global convergence statement, are documented in a companion technical note."* (The companion note exists at `analytics/math/argiento_obstruction.md` and remains in the repo as scaffolding.) The footnote was **dropped** from `main_v2.tex` on 2026-05-19 because the paper's `.tex` sources should not reference external markdowns; the surrounding prose already says "Whether some other route to a global convergence statement is available is genuinely open," which stands on its own. If you ever want the paper to refer to the obstruction analysis directly, draft it as an Appendix subsection rather than re-introducing the footnote.

## Compile and display

### Toolchain

TeX Live is installed at `/Library/TeX/texbin/`. `pdflatex`, `bibtex`, `latexmk` are all on `$PATH`. macOS default — no manual setup needed.

### One-shot compile

From `manuscript/` (you must `cd` here first, because `\graphicspath{}` is relative to the `.tex` file's directory):

```bash
cd manuscript/
latexmk -pdf main_v2.tex
open main_v2.pdf
```

`latexmk -pdf` runs `pdflatex → bibtex → pdflatex → pdflatex` automatically, stopping when nothing more changes. Subsequent compiles after small edits typically need only one `pdflatex` pass since `.aux` and `.bbl` are already up to date.

### First-pass gotcha

On a **cold** compile (no `.aux` / `.bbl` yet), `latexmk` exits non-zero on the first pass with messages like:

```
LaTeX Warning: Citation `grice1975logic' on page 1 undefined…
Latex failed to resolve 46 citation(s)
```

This is benign — bibtex hasn't run yet because pdflatex didn't yet have an `.aux` to read citations from. Re-run `latexmk -pdf main_v2.tex` immediately; the second invocation runs bibtex, then re-runs pdflatex, and exits clean.

Alternative (avoids the cold-start error): run the pipeline manually.

```bash
pdflatex -interaction=nonstopmode main_v2.tex
bibtex main_v2
pdflatex -interaction=nonstopmode main_v2.tex
pdflatex -interaction=nonstopmode main_v2.tex
```

### Smoke compile (no PDF output)

When iterating on the .tex without caring about the visual output (e.g. just want to know whether the LaTeX is syntactically valid and which figures are missing), use `-draftmode`:

```bash
pdflatex -interaction=nonstopmode -draftmode main_v2.tex 2>&1 | tail -60
```

This still writes the `.aux` / `.log` but skips PDF generation. Missing figures are reported as `File '<name>' not found` warnings without halting; you get the full error landscape on one pass.

### Display

On macOS: `open main_v2.pdf` opens the default PDF viewer (Preview). The viewer reloads automatically when the PDF is rebuilt — no need to close and reopen between compiles.

## Build artifacts

`pdflatex` + `bibtex` produce the following files in `manuscript/` alongside `main_v2.tex`:

| File | Purpose | Track in git? |
|---|---|---|
| `main_v2.pdf` | The compiled output | **Yes** (per the KB skill convention) |
| `main_v2.aux` | Citation + cross-reference list (read by bibtex and by subsequent pdflatex passes) | No |
| `main_v2.log` | Compile log | No |
| `main_v2.bbl` | Formatted bibliography (read by pdflatex from bibtex's output) | No |
| `main_v2.blg` | bibtex log | No |
| `main_v2.out` | Hyperref bookmarks | No |
| `main_v2.fls` | File list (read by latexmk) | No |
| `main_v2.fdb_latexmk` | latexmk build database | No |

**`.gitignore` status (as of 2026-05-18)**: a LaTeX block was added to the repo's `.gitignore` when the paper sources moved to `manuscript/`. The aux files above are now ignored everywhere in the tree:

```gitignore
# LaTeX build artifacts (manuscript/ is the canonical source tree)
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.synctex.gz
*.toc
*.lof
*.lot
```

(`*.pdf` is **not** in `.gitignore` — compiled PDFs alongside `main.tex` / `main_v2.tex` / `Appendix.tex` are intentionally tracked, per the KB skill's "Notes" section.)

## Gotchas

- **`References.bib` is large and shared across projects.** 257 entries, only ~28 cited from `main.tex`. Many "ghost" keys (Levi, Kyburg, Seidenfeld, Hansson, AGM-family, etc.) belong to unrelated papers in the user's library. Grep before adding; reuse before inventing.
- **Undefined citation keys render as `[?]`, silently.** `pdflatex` does not error on a missing `\cite{}` key — it just produces a literal `?`. If you compile and see `[?]` in the output, grep `References.bib` for the key (with the user's capitalization).
- **Inputenc + UTF-8 + literal `§`.** `main.tex` line 2 loads `\usepackage[utf8]{inputenc}` and prose freely contains the section symbol `§`. Don't replace `§` with `\S`; keep the literal character.
- **`mathptmx` + `mathabx`.** Loaded together (lines 19, 74). Some symbols are redefined by the later import. If a math glyph renders strangely after an edit, suspect `mathabx` first.
- **`numbers` natbib + `plainnat` style** is a slightly unusual combo (`plainnat` is the author-year default; with `[numbers]` it switches to numeric ordering). Don't "fix" this to `[authoryear]` or to a different `\bibliographystyle{}` without asking — the user may have set this up deliberately.
- **No `\label{}` on sections** means `\cleardoublepage` / `\appendix` page-break behavior is the only navigation aid. If you start needing programmatic cross-references between sections, raise the topic; don't unilaterally add labels.
- **Working directory matters for `\graphicspath{}`.** Always `cd manuscript/` before running `pdflatex` / `latexmk`. The relative paths in `\graphicspath{}` (`../results/...`) resolve relative to the *current working directory of the compile process*, not the .tex file's directory. Running pdflatex from the repo root will look for figures in `../results/...` from the repo root (which won't exist) and produce "file not found" for every figure.
- **`latexmk` exits non-zero on cold compiles.** See "First-pass gotcha" under Compile and display. Re-run; don't treat the first error as a real failure.
- **`replace_all` is the safe default for figure-name updates.** `main_v2.tex` keeps commented-out backup blocks of each figure environment (around lines 529–541 and 572–584) that mirror the active block. When renaming a figure, use `Edit` with `replace_all=true` so the active and backup blocks stay in sync — a backup that drifts from the live version is worse than no backup at all.
