# Verification scripts

- status: active
- type: how-to
- id: rl_signaling.analytics.scripts.readme
- description: How to run and interpret the independent verification scripts that cross-check the math files in analytics/ against the rl_signaling implementation. Each script uses scipy or hand-coded numpy as an independent reference, never trusting rl_signaling's own implementation as ground truth.
- label: [reference, math]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-09
<!-- content -->

This folder contains independent verification scripts for the math identities written down in [analytics/](..). Each script is self-contained and can be run individually. Five scripts (`verify_*.py`) cover the per-cell math (information theory, Q-learning, TD-learning, costly signaling, urn convergence); six study scripts (`study_*.py`, `enumerate_*.py`) cover the modeler-perspective Markov-chain analysis of §2.3.

```bash
.venv/bin/python -m analytics.scripts.verify_information_theory
.venv/bin/python -m analytics.scripts.verify_q_learning
.venv/bin/python -m analytics.scripts.verify_td_learning
.venv/bin/python -m analytics.scripts.verify_costly_signaling
.venv/bin/python -m analytics.scripts.verify_urn_convergence
.venv/bin/python -m analytics.scripts.study_toy_markov_chain
.venv/bin/python -m analytics.scripts.enumerate_absorbing_states
.venv/bin/python -m analytics.scripts.study_urn_basin_drift
.venv/bin/python -m analytics.scripts.study_factored_kernel
.venv/bin/python -m analytics.scripts.study_polya_signaling_convergence
.venv/bin/python -m analytics.scripts.study_coarse_grained_mle
```

Each script prints one PASS/FAIL line per check and exits with status code 0 if every check passed, non-zero otherwise.

## Verification posture

The pattern across all scripts is **two-source comparison**:

1. The script computes a quantity (entropy, Q-value, NMI, urn probability, etc.) using an **independent** implementation — usually `scipy.stats.entropy`, hand-coded numpy summations, or analytical closed-form formulas derived in the corresponding math file.
2. The script computes the **same** quantity using `rl_signaling` (or by running an agent through the env).
3. The script asserts the two agree to a documented tolerance — usually `atol = 1e-12` for finite-step cases, looser for asymptotic / convergence cases.

This is more rigorous than the unit tests under [tests/](../../tests/), which compare `rl_signaling` against values derived **from the same math** that produced the implementation. If a docstring derivation is wrong, both the docstring and the test could agree against a wrong target. The scripts close that gap by using a different implementation as the reference.

## Why scripts and not tests

Three reasons:

1. **Pedagogical.** The scripts are tutorials. They are read while reading the math, with explicit comments tying each numerical check to a section of the corresponding `.md` file. Tests under [tests/](../../tests/) are terse and not designed for that.
2. **Standalone.** Each script can be run in isolation while editing the corresponding math file, without invoking the rest of the test suite. This is faster than a full `pytest tests/` run.
3. **Independent dependency.** Tests under `tests/` constrain the package to dependencies declared in `pyproject.toml`'s `[project.optional-dependencies] dev` extra. Scripts here can use additional reference dependencies (`scipy`, `sympy`, etc.) without forcing them on the test extra.

## Script-by-script summary

| Script | Math file | Reference implementation |
|---|---|---|
| `verify_information_theory.py` | [information_theory.md](../math/information_theory.md) | `scipy.stats.entropy` |
| `verify_q_learning.py` | [agent_q_learning.md](../math/agent_q_learning.md) | Closed-form $Q_n = r(1 - (1-\alpha)^n)$ + asymptotic variance |
| `verify_td_learning.py` | [agent_td_learning.md](../math/agent_td_learning.md) | Empirical mean (Robbins–Monro), bootstrap algebra |
| `verify_costly_signaling.py` | [costly_signaling.md](../math/costly_signaling.md) | Hand arithmetic via `MultiAgentEnv.reward` |
| `verify_urn_convergence.py` | [agent_urn.md](../math/agent_urn.md) | Closed-form sampling probability + empirical Monte Carlo |
| `study_toy_markov_chain.py` | [proof_of_concept_markov.md](../math/proof_of_concept_markov.md), [initialization_basins.md](../math/initialization_basins.md) | Closed-form recursion for $\rho_t$ + 50k-trajectory MC |
| `enumerate_absorbing_states.py` | [proof_of_concept_markov.md](../math/proof_of_concept_markov.md) | Brute-force enumeration of all 2304 deterministic policy profiles + reward computation |
| `study_urn_basin_drift.py` | [proof_of_concept_markov.md](../math/proof_of_concept_markov.md), [initialization_basins.md](../math/initialization_basins.md) | UrnAgent only (Roth-Erev): 200-seed MC at `[1, 0]` validates analytical absorbing-state distribution; per-init drift snapshots across `init_weights`. Q-learning analysis is deferred — see `TODO_WORKFLOW.md::todo.qlearning_proof_of_concept`. |
| `study_factored_kernel.py` | [docs/roth_erev_polya_mle.md](../math/roth_erev_polya_mle.md) §2, §6 | Ports the doc's `one_step_kernel_value` to the simulator's dict-of-array urns; validates choice rule (100k MC), single-urn transition factorization vs `P(x) · n_σ/S · q*(x)`, and full-state kernel sum = 1 across all 256 candidate next states. |
| `study_polya_signaling_convergence.py` | [docs/roth_erev_polya_mle.md](../math/roth_erev_polya_mle.md) §3, [proof_of_concept_markov.md](../math/proof_of_concept_markov.md) §"Pure-Pólya signaling-urn convergence" | Empirical validation of the Dirichlet limit theorem under a frozen partner and frozen action policy: KS test against `Beta(n_0)` marginal across M = 200 seeds at T = 8000 episodes. |
| `study_coarse_grained_mle.py` | [docs/roth_erev_polya_mle.md](../math/roth_erev_polya_mle.md) §4–5 | Coarse-grained MLE: empirical transition matrices on modal-map, simplex-bin, and NMI-bin projections. Basin-reach probabilities at NMI > 0.7 / 0.9 thresholds. Visit-time fraction in the high-NMI basin is monotone non-decreasing in pre-seed strength, confirming the §2.3 informal claim. |

## Common conventions

- All scripts use `numpy` and `numpy.random` for randomness, with a fixed seed for reproducibility (printed in the output).
- Tolerances are `atol` (absolute) for exact-arithmetic cases, `rtol` (relative) for asymptotic cases.
- Output uses simple plain-text PASS/FAIL lines for grep-ability:
  ```
  [PASS] H(uniform_2) = 1.0 bits — rl_signaling=1.0, scipy=1.0, diff=0.0
  [PASS] H(uniform_4) = 2.0 bits — rl_signaling=2.0, scipy=2.0, diff=0.0
  ```
- No external file I/O. Scripts compute, assert, print, exit. They do not produce CSVs or PNGs.

## Running all at once

```bash
for s in verify_information_theory verify_q_learning verify_td_learning \
         verify_costly_signaling verify_urn_convergence \
         study_toy_markov_chain enumerate_absorbing_states \
         study_urn_basin_drift study_factored_kernel \
         study_polya_signaling_convergence study_coarse_grained_mle; do
    echo "=== $s ===";
    .venv/bin/python -m analytics.scripts.$s || echo "[FAIL: $s]";
done
```

The `study_polya_signaling_convergence` and `study_coarse_grained_mle` scripts each take ~10–40 s; the rest finish in a few seconds. Total wall time for the full loop is roughly one minute on the machine the project was developed on.

A future `run_all.py` could wrap this loop. Not currently implemented; the one-liner above is sufficient.
