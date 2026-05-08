# Analytics — Mathematical Reference

- status: active
- type: reference
- id: rl_signaling.analytics.analytics_readme
- description: Index and conventions for the analytics/ folder — exhaustive mathematical descriptions of every quantity computed by the rl_signaling codebase, plus independent verification scripts.
- label: [reference, math]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-08
<!-- content -->

This folder is the project's mathematical reference. It defines every quantity the [rl_signaling/](../rl_signaling/) package computes, derives the identities the code depends on, and provides numerical worked examples. The accompanying [scripts/](scripts/) verify the math by running **independent** implementations (numpy / scipy) against the package and asserting agreement.

The companion documents are:

- [DEBUGGING_PLAN.md](../DEBUGGING_PLAN.md) — Phase 1 confirmed model specification (the *what*).
- [MODELING_CHOICES_REF.md](../MODELING_CHOICES_REF.md) — design-space catalog (the *why this choice*).
- This folder — the math behind those choices and what the code actually computes (the *how*, formally).

## Reading order

Read top-to-bottom; each file builds on the previous.

| # | File | Topic | Depends on |
|---|---|---|---|
| 0 | [notation.md](notation.md) | Symbols, sets, indexing conventions | — |
| 1 | [information_theory.md](information_theory.md) | Shannon entropy, mutual information, NMI | notation |
| 2 | [signaling_model.md](signaling_model.md) | World state, observations, signals, payoff | notation |
| 3 | [costly_signaling.md](costly_signaling.md) | Null signal, cost flow | signaling_model |
| 4 | [agent_urn.md](agent_urn.md) | Roth–Erev urn dynamics | signaling_model |
| 5 | [agent_q_learning.md](agent_q_learning.md) | Single-step TD, exploration decay | signaling_model |
| 6 | [agent_td_learning.md](agent_td_learning.md) | Bootstrap, count-based α, two-phase update | signaling_model, agent_q_learning |
| 7 | [exploration_strategies.md](exploration_strategies.md) | ε-greedy, softmax, UCB | agent_q_learning |
| 8 | [metrics_aggregation.md](metrics_aggregation.md) | Trajectory → CSV column → figure pipeline; producer/consumer trace | all of the above |
| — | [scripts/](scripts/) | Independent verification scripts | all of the above |

## Conventions

- **Math notation.** GitHub-flavored markdown with native LaTeX (`$inline$`, `$$display$$`). Identifiers in code are rendered with backticks (e.g. `q_table`).
- **Log base.** All entropies and mutual informations are in **bits**, i.e. base 2. The code uses `np.log2`. See [information_theory.md](information_theory.md) for the full convention.
- **Indexing.** Agent index $i \in \{0, 1, \dots, N-1\}$ (zero-based, matching Python). Episode index $t = 1, 2, \dots$ (one-based when discussing time, zero-based when slicing arrays — disambiguated explicitly where it matters).
- **Tuples vs vectors.** State and observation values are written as tuples $\mathbf{v} = (v_1, \dots, v_n)$. Distributions are written as row vectors $\mathbf{p} = (p_1, \dots, p_k)$.
- **Probability of an event.** $\mathbb{P}[\cdot]$ for events; $p(\cdot)$ for probability mass functions. $\mathbb{E}[\cdot]$ for expectation.
- **Code references.** When a formula maps to a specific code line, the file uses the form [rl_signaling/agents.py:447](../rl_signaling/agents.py#L447) so the reader can jump from the math to the implementation.

## Verification posture

Every non-trivial identity in this folder is checked at least once by either:

1. A unit test in [tests/](../tests/) (most are in [tests/test_numerical_sanity.py](../tests/test_numerical_sanity.py) and [tests/test_info_theory.py](../tests/test_info_theory.py)).
2. A standalone script in [scripts/](scripts/) that uses an independent reference implementation (e.g. `scipy.stats.entropy`, hand-coded summations) and asserts agreement with `rl_signaling`.

The two-source pattern is intentional. A single-source check (the code matches itself) cannot detect a derivation error in the docstring; a two-source check (code matches an independent reference) can. The scripts under [scripts/](scripts/) are the second source.

## How to run the scripts

```bash
.venv/bin/python -m analytics.scripts.verify_information_theory
.venv/bin/python -m analytics.scripts.verify_q_learning
.venv/bin/python -m analytics.scripts.verify_td_learning
.venv/bin/python -m analytics.scripts.verify_costly_signaling
.venv/bin/python -m analytics.scripts.verify_urn_convergence
```

Each script prints a one-line PASS/FAIL summary per check and exits non-zero if any check failed. They are deliberately not `pytest` tests — they are tutorials that double as regression checks, and you can run them individually while reading the corresponding math file.

## File metadata schema

Every `.md` file in this folder follows the metadata format used elsewhere in the repository (see [LEGACY_BUGS_LOG.md](../LEGACY_BUGS_LOG.md), [DEBUGGING_PLAN.md](../DEBUGGING_PLAN.md), etc.):

```markdown
# Title
- status: active
- type: explanation | reference
- id: rl_signaling.analytics.<slug>
- description: one-line summary
- label: [reference, math]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: YYYY-MM-DD
<!-- content -->
```
