# Publication Checklist
- status: active
- type: workflow
- id: rl_signaling.publication_checklist
- description: Record of the pre-publication history scrub performed on 2026-08-12 — what was removed and why, where the backup lives — plus the remaining steps before this repository is made public.
- label: [core]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-08-12
<!-- content -->

This repository accompanies **"Signaling Games with Distributed Rewards"** (PHOS-17993), accepted at *Philosophy of Science*. It is **code-only by design**: the manuscript, the referee correspondence, and the talk slides are deliberately absent, from the working tree *and* from every commit in history.

## Why the repository is code-only

| Concern | What it excludes |
|:---|:---|
| **Journal copyright** | The accepted and published versions of the article belong to the publisher. `manuscript/` is not distributed here. |
| **Referee confidentiality** | Referee reports and response letters are confidential correspondence between author, editor, and reviewers. |
| **Scope** | Slides, submission-process notes, and internal audit logs are not part of the scientific record this repository exists to support. |

These files still exist **on the author's disk** at their usual paths. They are listed in `.gitignore` under a "Publication boundary" block. Do not `git add -f` them.

## What was removed, and when

On **2026-08-12** the history was rewritten with `git filter-repo --invert-paths`, rewriting all 3 branches. Untracking alone was insufficient: the material had already been committed, so it would have remained readable in historical commits.

| Path | Reason |
|:---|:---|
| `manuscript/` | Manuscript sources, compiled PDFs, referee reports, response letters, submitted PDFs |
| `analytics/docs/` | The manuscript's **former location** before the 2026-05 layout migration — including `Response_to_Reviewers_Revised.docx`. Easy to miss; scrubbing only `manuscript/` would have left it behind. |
| `slides/` | Talk slides built on the paper |
| `plots_and_results/` | Pre-refactor data directory, ~20 MB of CSVs superseded by `results/` |
| `docs/code-audit/`, `docs/JOURNAL_WORD_LIMIT.md`, `docs/Gmail*.pdf` | Internal audit trail and submission-process correspondence |
| `writing_comments.md` | Raw writing feedback |
| `.claude/` | Agent tooling config — local absolute paths and a detailed record of the editing workflow |
| `worklog.jsonl`, `WORKLOG.md`, `TODO_WORKFLOW.md` | Per-repo governance artifacts, retired 2026-07-31 in favour of the central store |
| `dummy_plot.png` | Default `file_path` fallback in `plotting.py`; scratch output |
| `**/__pycache__/`, `**/.DS_Store`, `*.json-e` | Build and editor junk committed before `.gitignore` covered it |

Result: **226 → 148 tracked files**, `.git` **288 MB → 109 MB**.

A scan of full history found **no credentials, `.env` files, keys, or tokens** at any commit.

## Backup

A complete pre-scrub backup was taken before the rewrite and verified (`git bundle verify` → *"The bundle records a complete history"*):

```
~/Desktop/RL_Signaling_backups/
  RL_Signaling-7f770cc.bundle     # 152 MB, all refs, pre-scrub
  RL_Signaling-mirror.git/        # full mirror clone
```

To inspect or recover anything from the old history:

```bash
git clone ~/Desktop/RL_Signaling_backups/RL_Signaling-7f770cc.bundle recovered/
```

> **This backup is now load-bearing — do not delete it.** The old remote was deleted and recreated on 2026-08-12, and the pre-scrub commits are confirmed gone from GitHub. Your working copies of `manuscript/` and `slides/` are still on disk, but **their git history now exists nowhere else.** Move this backup to durable storage (external drive or encrypted cloud); a single Desktop folder is not a backup.

## Remaining steps before going public

- [x] **Delete the existing GitHub remote and recreate it empty.** *(Done 2026-08-12.)* Force-pushing rewritten history leaves the old objects in GitHub's storage, reachable by direct SHA until their garbage collection runs. Deleting the repository was the immediate guarantee.
- [x] **Push all branches to the fresh remote.** *(Done 2026-08-12.)*
- [x] **Re-clone and verify.** *(Done 2026-08-12.)* A fresh clone from GitHub shows none of the excluded paths in any of 227 commits across all three branches, the three pre-scrub SHAs (`7f770cc`, `2cb5ab3`, `104868b9`) are unreachable, and the test suite passes (80 tests) from the clone.
- [ ] **Move the backup to durable storage** — see the warning above.
- [ ] **Add the article citation and DOI** to `README.md` once available.
- [ ] **Check the journal's data/code policy** for a required deposit (Zenodo, OSF) and mint a DOI for the repository if expected.
- [ ] **Decide on preprint linkage.** The accepted manuscript cannot be distributed here, but *Philosophy of Science* policy typically permits an author-accepted-manuscript postprint elsewhere; link it from the README rather than committing it.
- [ ] **Flip the repository to public.**

## If the manuscript ever needs to be added back

It should not be. If policy changes and the *published* version becomes distributable, add it as a **release asset** or a link to the publisher's DOI rather than committing it — that keeps the copyright-bearing artifact out of the git history, where it cannot be selectively removed later without another rewrite.
