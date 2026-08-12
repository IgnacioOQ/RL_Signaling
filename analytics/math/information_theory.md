# Information theory

- status: active
- type: explanation
- id: rl_signaling.analytics.information_theory
- description: Shannon entropy, conditional entropy, joint entropy, mutual information, and normalized mutual information as used by rl_signaling/info_theory.py — definitions, identities, derivations, edge cases, and worked numerical examples.
- label: [reference, math]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-08
<!-- content -->

This file develops the information-theoretic quantities the project uses to measure how much information a signal carries about an agent's observation. The metrics are computed by [rl_signaling/info_theory.py](../../rl_signaling/info_theory.py) and consumed by every notebook that records "NMI" as a column in a results CSV.

The intended interpretation throughout: $O$ is the agent's **observation** of the world state; $S$ is the **signal** the agent emits in response. NMI measures how much of the observation's entropy is explained by the signal — equivalently, "how informative is the signal about what the agent saw?".

## Probability primer

Let $X$ be a discrete random variable taking values in a finite alphabet $\mathcal{X}$. Its **probability mass function** (PMF) is

$$p(x) := \mathbb{P}[X = x], \qquad x \in \mathcal{X},$$

with $\sum_{x \in \mathcal{X}} p(x) = 1$ and $p(x) \ge 0$ for all $x$. The **support** of $X$ is

$$\operatorname{supp}(X) := \{ x \in \mathcal{X} : p(x) > 0 \}.$$

For a pair $(X, Y)$ with joint PMF $p(x, y)$, the **marginals** are

$$p_X(x) = \sum_{y \in \mathcal{Y}} p(x, y), \qquad p_Y(y) = \sum_{x \in \mathcal{X}} p(x, y),$$

and the **conditional** of $X$ given $Y = y$ (defined only when $p_Y(y) > 0$) is

$$p_{X \mid Y}(x \mid y) := \frac{p(x, y)}{p_Y(y)}.$$

These are the only probability tools needed for the rest of this file.

## Shannon entropy

### Definition

For a discrete random variable $X$ with PMF $p$:

$$\boxed{\; H(X) \;:=\; -\sum_{x \in \operatorname{supp}(X)} p(x) \log_2 p(x) \;}$$

The base of the logarithm fixes the unit:

- **base 2** → bits (this project)
- base $e$ → nats
- base 10 → bans

The convention $0 \log 0 := 0$ is used implicitly by restricting the sum to $\operatorname{supp}(X)$. This is justified by

$$\lim_{p \to 0^+} p \log p = 0.$$

### Properties

1. **Non-negativity.** $H(X) \ge 0$, with $H(X) = 0$ iff $X$ is deterministic (the support has size 1).
2. **Maximum at uniform.** Among distributions on $\lvert\mathcal{X}\rvert = n$ elements, the maximum of $H$ is attained at the uniform distribution and equals $\log_2 n$.
3. **Concavity.** $H$ is concave in $p$.
4. **Continuity.** $H$ is continuous in $p$ (with the $0\log 0 = 0$ convention).

### Implementation

The package's helper [_compute_entropy](../../rl_signaling/info_theory.py#L14-L16) implements the definition above:

```python
def _compute_entropy(probabilities):
    return -sum(p * np.log2(p) for p in probabilities if p > 0)
```

The `if p > 0` filter encodes the $0 \log 0 = 0$ convention.

### Worked example

For $X$ uniform on two values, $p = (1/2, 1/2)$:

$$H(X) = -\Big(\tfrac{1}{2} \log_2 \tfrac{1}{2} + \tfrac{1}{2} \log_2 \tfrac{1}{2}\Big) = -\Big(\tfrac{1}{2} \cdot (-1) + \tfrac{1}{2} \cdot (-1)\Big) = 1 \text{ bit.}$$

For $X$ uniform on four values, $p = (1/4, 1/4, 1/4, 1/4)$:

$$H(X) = -4 \cdot \tfrac{1}{4} \log_2 \tfrac{1}{4} = \log_2 4 = 2 \text{ bits.}$$

These two equalities are the test cases in [tests/test_numerical_sanity.py::test_entropy_is_in_bits_log_base_2](../../tests/test_numerical_sanity.py#L23-L33).

## Joint entropy

For a pair $(X, Y)$ with joint PMF $p(x, y)$:

$$H(X, Y) := -\sum_{x, y \in \operatorname{supp}(X, Y)} p(x, y) \log_2 p(x, y).$$

This is just the entropy of the product variable $(X, Y)$ treated as a single random variable on $\mathcal{X} \times \mathcal{Y}$.

## Conditional entropy

### Definition

The **conditional entropy** of $X$ given $Y$ is the expected entropy of the conditional distribution $p_{X \mid Y}$:

$$\boxed{\; H(X \mid Y) \;:=\; \sum_{y \in \operatorname{supp}(Y)} p_Y(y) \cdot H(X \mid Y = y) \;}$$

where $H(X \mid Y = y) := -\sum_{x} p_{X \mid Y}(x \mid y) \log_2 p_{X \mid Y}(x \mid y)$.

### Identity (chain rule)

$$H(X, Y) = H(Y) + H(X \mid Y) = H(X) + H(Y \mid X).$$

**Derivation.**

$$\begin{aligned}
H(X, Y) &= -\sum_{x,y} p(x,y) \log_2 p(x,y) \\
        &= -\sum_{x,y} p(x,y) \log_2 \big( p_Y(y) \cdot p_{X|Y}(x|y) \big) \\
        &= -\sum_{x,y} p(x,y) \log_2 p_Y(y) \;-\; \sum_{x,y} p(x,y) \log_2 p_{X|Y}(x|y) \\
        &= -\sum_{y} p_Y(y) \log_2 p_Y(y) \;+\; \sum_{y} p_Y(y) \cdot \Big( -\sum_x p_{X|Y}(x|y) \log_2 p_{X|Y}(x|y) \Big) \\
        &= H(Y) + H(X \mid Y).
\end{aligned}$$

The third line splits the log of a product. The fourth uses $\sum_x p(x, y) = p_Y(y)$ for the first term and $p(x,y) = p_Y(y) p_{X|Y}(x|y)$ for the second.

### Properties

1. **Non-negativity.** $H(X \mid Y) \ge 0$, with equality iff $X$ is a deterministic function of $Y$.
2. **Conditioning reduces entropy.** $H(X \mid Y) \le H(X)$, with equality iff $X$ and $Y$ are independent.

## Mutual information

### Definition

For random variables $X, Y$:

$$\boxed{\; I(X ; Y) \;:=\; H(X) + H(Y) - H(X, Y). \;}$$

### Equivalent forms

By the chain rule above, $H(X, Y) = H(Y) + H(X \mid Y)$, so

$$I(X ; Y) = H(X) - H(X \mid Y).$$

By symmetry (chain rule applied the other way),

$$I(X ; Y) = H(Y) - H(Y \mid X).$$

A third form uses the joint and marginal PMFs directly:

$$I(X ; Y) = \sum_{x, y} p(x, y) \log_2 \frac{p(x, y)}{p_X(x) \, p_Y(y)}.$$

(Substituting the chain-rule expansion into the entropy definitions reproduces this — Cover & Thomas, *Elements of Information Theory*, §2.3.)

### Properties

1. **Non-negativity.** $I(X ; Y) \ge 0$, with equality iff $X \perp Y$ (independence). This is a consequence of Jensen's inequality / Gibbs' inequality.
2. **Symmetry.** $I(X ; Y) = I(Y ; X)$.
3. **Self-information.** $I(X ; X) = H(X)$.
4. **Bound.** $I(X ; Y) \le \min\{ H(X), H(Y) \}$.

### Implementation

The package implements [compute_mutual_information](../../rl_signaling/info_theory.py#L19-L61) using the form $I(S; O) = H(S) - H(S \mid O)$:

```python
H_S = _compute_entropy(P_S.values())
H_S_given_O = sum(P_O[o] * _compute_entropy(per_o_dist) for o, per_o_dist in ...)
I_S_O = H_S - H_S_given_O
```

The naming convention reverses the convention used in this file: in the code, $S$ is the **signal** and $O$ is the **observation**, so the input to the function is `agent_signal_usage = {observation -> signal counts}` (a "$O \to S$" mapping). The math is identical because mutual information is symmetric.

## Normalized mutual information (NMI)

### The need for normalization

$I(X; Y)$ has units of bits. Its magnitude depends on the alphabets — for two random variables on $n$ atoms each, $I$ ranges over $[0, \log_2 n]$. To compare across experiments with different alphabets, we normalize.

### Variants in the literature

Three normalizations are common; this project uses the first.

**Asymmetric (output-side) — used by this project.**

$$\mathrm{NMI}(X; Y) := \frac{I(X; Y)}{H(Y)} = 1 - \frac{H(Y \mid X)}{H(Y)}.$$

Range: $[0, 1]$. Interprets as "fraction of $Y$'s uncertainty explained by $X$." The denominator should be the "output" or "receiving" side. In the project the signal is the *output* of the encoding policy, so $Y$ is the signal and the denominator is $H(\text{signal})$ — but the input to `compute_mutual_information` is `agent_signal_usage` keyed by *observation*, and the code divides by the entropy of the *observation* marginal. That naming inversion (code calls "$O$" what the project's spec calls "the side we normalize by") is a docstring-level detail; the math is symmetric.

**Geometric.**

$$\mathrm{NMI}_{\text{geom}}(X; Y) := \frac{I(X; Y)}{\sqrt{H(X) \cdot H(Y)}}.$$

Range: $[0, 1]$. Symmetric. Less common in signaling-game contexts but standard in clustering literature (e.g. scikit-learn's `normalized_mutual_info_score(..., average_method='geometric')`).

**Arithmetic.**

$$\mathrm{NMI}_{\text{arith}}(X; Y) := \frac{2 I(X; Y)}{H(X) + H(Y)}.$$

Range: $[0, 1]$. Symmetric. Also called the "redundancy."

The project's choice of asymmetric output-side normalization is locked in by Axis 21 of the Phase 1 confirmed model specification.

### Edge case: zero-entropy denominator

If $H(Y) = 0$ — meaning $Y$ is constant — the asymmetric NMI is undefined ($0/0$). The project's convention (Axis 22) is

$$\mathrm{NMI}(X; Y) := 0 \quad \text{when} \quad H(Y) = 0.$$

Implementation: the line `NMI = I_S_O / H_O if H_O > 0 else 0` at [rl_signaling/info_theory.py:59](../../rl_signaling/info_theory.py#L59).

### Empirical estimation

In practice, the package never has access to the true joint PMF $p(x, y)$. It has empirical counts: `agent_signal_usage[obs][signal_index] = count` accumulated over episodes. The empirical PMF estimator is the maximum-likelihood plug-in:

$$\hat{p}(x, y) := \frac{n_{x, y}}{\sum_{x', y'} n_{x', y'}}.$$

All entropies and mutual informations in the code are computed from $\hat{p}$. This is a **biased** estimator of the true MI for small sample sizes (it tends to overestimate); the bias decays as $O(1/n)$ where $n$ is the total sample count, and is negligible for the sample sizes used in the project's experiments (typically $n \ge 1000$).

A bias-corrected estimator (Miller-Madow, Grassberger, James-Stein) would shrink the empirical entropy upward. Not used here — accepted approximation.

## Worked numerical example: NMI = 1 by hand

Consider the 2x2 signal-usage table

$$\begin{pmatrix} 10 & 0 \\ 0 & 10 \end{pmatrix}$$

read as: when the agent observed $o = (0,)$ it emitted signal 0 ten times and signal 1 zero times; when it observed $o = (1,)$ it emitted signal 0 zero times and signal 1 ten times.

**Marginals.** Total samples $n = 20$.

$$\hat{p}_O(0) = \tfrac{10 + 0}{20} = \tfrac{1}{2}, \quad \hat{p}_O(1) = \tfrac{0 + 10}{20} = \tfrac{1}{2}.$$

$$\hat{p}_S(0) = \tfrac{10 + 0}{20} = \tfrac{1}{2}, \quad \hat{p}_S(1) = \tfrac{0 + 10}{20} = \tfrac{1}{2}.$$

**Marginal entropies.**

$$H(O) = H(S) = -2 \cdot \tfrac{1}{2} \log_2 \tfrac{1}{2} = 1 \text{ bit.}$$

**Conditional distributions.**

$$\hat{p}_{S \mid O}(\cdot \mid 0) = (1, 0), \qquad \hat{p}_{S \mid O}(\cdot \mid 1) = (0, 1).$$

Each is one-hot, so $H(S \mid O = o) = 0$ for both $o$, giving

$$H(S \mid O) = \hat{p}_O(0) \cdot 0 + \hat{p}_O(1) \cdot 0 = 0.$$

**Mutual information.**

$$I(S ; O) = H(S) - H(S \mid O) = 1 - 0 = 1 \text{ bit.}$$

**NMI.**

$$\mathrm{NMI}(S; O) = \frac{I(S; O)}{H(O)} = \frac{1}{1} = 1.$$

This is the test case in [tests/test_numerical_sanity.py::test_perfect_2x2_correlation_nmi_is_one_by_hand](../../tests/test_numerical_sanity.py#L36-L51).

## Worked numerical example: NMI = 0 (independence)

Consider the 2x2 table

$$\begin{pmatrix} 5 & 5 \\ 5 & 5 \end{pmatrix}.$$

Marginals: $\hat{p}_O = \hat{p}_S = (1/2, 1/2)$. Conditional distributions: $\hat{p}_{S \mid O}(\cdot \mid 0) = \hat{p}_{S \mid O}(\cdot \mid 1) = (1/2, 1/2)$.

$$H(S \mid O) = \tfrac{1}{2} \cdot 1 + \tfrac{1}{2} \cdot 1 = 1 \text{ bit} = H(S).$$

$$I(S ; O) = H(S) - H(S \mid O) = 1 - 1 = 0.$$

$\mathrm{NMI} = 0/1 = 0$. The signal carries no information about the observation. Test: [tests/test_info_theory.py::test_independence_gives_zero_nmi](../../tests/test_info_theory.py#L22-L30).

## Cross-references

| Quantity | Definition | Code | Test |
|---|---|---|---|
| Shannon entropy $H$ | this file, "Shannon entropy" | [rl_signaling/info_theory.py:14-16](../../rl_signaling/info_theory.py#L14-L16) | [test_numerical_sanity.py::test_entropy_is_in_bits_log_base_2](../../tests/test_numerical_sanity.py#L23-L33) |
| Mutual information $I$ | this file, "Mutual information" | [rl_signaling/info_theory.py:19-61](../../rl_signaling/info_theory.py#L19-L61) | [test_info_theory.py](../../tests/test_info_theory.py), [test_numerical_sanity.py::test_perfect_2x2_correlation_nmi_is_one_by_hand](../../tests/test_numerical_sanity.py#L36-L51) |
| NMI (asymmetric) | this file, "Normalized mutual information" | [rl_signaling/info_theory.py:59](../../rl_signaling/info_theory.py#L59) | [test_info_theory.py](../../tests/test_info_theory.py) |
| $H(O) = 0$ → NMI := 0 | this file, "Edge case" | [rl_signaling/info_theory.py:59](../../rl_signaling/info_theory.py#L59) | [test_info_theory.py::test_single_observation_gives_zero_nmi](../../tests/test_info_theory.py#L33-L37) |

## Independent verification

The script [scripts/verify_information_theory.py](scripts/verify_information_theory.py) re-derives every identity in this file using `scipy.stats.entropy` (a fully independent implementation) and asserts agreement with `rl_signaling.info_theory` to machine precision. Run with:

```bash
.venv/bin/python -m analytics.scripts.verify_information_theory
```
