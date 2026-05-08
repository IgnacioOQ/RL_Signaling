# Verification scripts

- status: active
- type: how-to
- id: rl_signaling.analytics.scripts.readme
- description: How to run and interpret the independent verification scripts that cross-check the math files in analytics/ against the rl_signaling implementation. Each script uses scipy or hand-coded numpy as an independent reference, never trusting rl_signaling's own implementation as ground truth.
- label: [reference, math]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-08
<!-- content -->

This folder contains five Python scripts that independently verify the math identities written down in [analytics/](..). Each script is self-contained and can be run individually:

```bash
.venv/bin/python -m analytics.scripts.verify_information_theory
.venv/bin/python -m analytics.scripts.verify_q_learning
.venv/bin/python -m analytics.scripts.verify_td_learning
.venv/bin/python -m analytics.scripts.verify_costly_signaling
.venv/bin/python -m analytics.scripts.verify_urn_convergence
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
| `verify_information_theory.py` | [information_theory.md](../information_theory.md) | `scipy.stats.entropy` |
| `verify_q_learning.py` | [agent_q_learning.md](../agent_q_learning.md) | Closed-form $Q_n = r(1 - (1-\alpha)^n)$ + asymptotic variance |
| `verify_td_learning.py` | [agent_td_learning.md](../agent_td_learning.md) | Empirical mean (Robbins–Monro), bootstrap algebra |
| `verify_costly_signaling.py` | [costly_signaling.md](../costly_signaling.md) | Hand arithmetic via `MultiAgentEnv.reward` |
| `verify_urn_convergence.py` | [agent_urn.md](../agent_urn.md) | Closed-form sampling probability + empirical Monte Carlo |

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
for s in verify_information_theory verify_q_learning verify_td_learning verify_costly_signaling verify_urn_convergence; do
    echo "=== $s ===";
    .venv/bin/python -m analytics.scripts.$s || echo "[FAIL: $s]";
done
```

A future `run_all.py` could wrap this loop. Not currently implemented; one-liner above is sufficient.
