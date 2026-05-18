# Estimating Transition Probabilities in the Roth–Erev Signal-Trading Markov Chain

## A modeler-side companion to §2.3 of *Signaling Games with Distributed Rewards*

This note works out, from a modeler's perspective, how to think about the
transition probabilities of the Markov chain defined in §2.3 of the paper, *for
the Roth–Erev case specifically*. Two questions are answered:

1. **What is the exact one-step transition probability of the chain?** This is
   the "Pólya-urn-style explicit computation": the kernel factors as a product
   of urn-sampling probabilities and reinforcement probabilities, every factor
   has a closed form, and you do not need to estimate anything to write down
   the kernel.
2. **What can you actually estimate by maximum likelihood?** The full state is
   non-recurrent, so naive transition counting fails. The right object is the
   chain *projected* onto a discrete feature, where plain MLE counting gives a
   meaningful and useful matrix.

The notation follows the paper: an agent has two tables `f[obs]` (signal
propensities) and `g[(obs, sig)]` (action propensities), both stored as
non-negative integer vectors. Choice probabilities are propensity-proportional;
updates are additive: `prop[chosen] += r`, with `r ∈ {0, 1}` in the matching
games of §2.

---

## 1. The state of the chain

Let $s_t$ be the full description of the system after episode $t$:

$$
s_t \;=\; \big(\, x_t,\, y_t,\; f^{(1)}_t,\, f^{(2)}_t,\; g^{(1)}_t,\, g^{(2)}_t,\;
                \sigma^{(1)}_t,\, \sigma^{(2)}_t,\; a^{(1)}_t,\, a^{(2)}_t,\;
                r^{(1)}_t,\, r^{(2)}_t \,\big).
$$

Three structural facts about this state space matter for everything below:

- **Discrete.** With integer-valued initializations and `r ∈ {0, 1}`, every
  propensity stays in $\mathbb{N}$. The state lives on an integer lattice, not
  a continuum.
- **Unbounded.** Propensities only ever go up (rewards are non-negative, and
  the version of Roth–Erev used in the paper has no decay term). After $T$
  episodes, the total mass in any single urn lies in $\{S_0, \dots, S_0 + T\}$.
- **Non-recurrent.** Because the lattice is unbounded and propensities are
  monotone, the chain almost surely never revisits any specific full state.
  Within a long simulation you observe each $s_t$ at most once.

This last point is what makes the naive MLE — counting transitions and
row-normalizing — meaningless on the full state.

---

## 2. The factored transition kernel

The paper notes that the kernel factors "sequentially by the product of
probabilities: nature's underlying state, signal choice, action choice, games
rewards, and learning update." Spelled out, with $\theta_t = (f^{(1)}_t,
f^{(2)}_t, g^{(1)}_t, g^{(2)}_t)$ denoting the urn state:

$$
\begin{aligned}
P(s_{t+1} \mid s_t) \;=\;
& \underbrace{P(x_{t+1})\,P(y_{t+1})}_{\text{nature}} \\
\times\;& \underbrace{\tfrac{f^{(1)}_t[x_{t+1}, \sigma^{(1)}_{t+1}]}{\sum_{\sigma'} f^{(1)}_t[x_{t+1}, \sigma']}}_{P(\sigma^{(1)}_{t+1}\mid x_{t+1},\, f^{(1)}_t)}
   \cdot \underbrace{\tfrac{f^{(2)}_t[y_{t+1}, \sigma^{(2)}_{t+1}]}{\sum_{\sigma'} f^{(2)}_t[y_{t+1}, \sigma']}}_{P(\sigma^{(2)}_{t+1}\mid y_{t+1},\, f^{(2)}_t)} \\
\times\;& \underbrace{\tfrac{g^{(1)}_t[(x_{t+1}, \sigma^{(2)}_{t+1}), a^{(1)}_{t+1}]}{\sum_{a'} g^{(1)}_t[(x_{t+1}, \sigma^{(2)}_{t+1}), a']}}_{P(a^{(1)}_{t+1}\mid \cdots)}
   \cdot \underbrace{\tfrac{g^{(2)}_t[(y_{t+1}, \sigma^{(1)}_{t+1}), a^{(2)}_{t+1}]}{\sum_{a'} g^{(2)}_t[(y_{t+1}, \sigma^{(1)}_{t+1}), a']}}_{P(a^{(2)}_{t+1}\mid \cdots)} \\
\times\;& \underbrace{\mathbf{1}\!\big[r^{(i)}_{t+1} = G_i(a^{(i)}_{t+1}, x_{t+1}, y_{t+1})\big]}_{\text{rewards: deterministic}} \\
\times\;& \underbrace{\mathbf{1}\!\big[\theta_{t+1} = \text{update}(\theta_t, \cdots)\big]}_{\text{updates: deterministic}}.
\end{aligned}
$$

Two observations:

- **The only stochastic factors are nature and the two pairs of urn draws.**
  Everything else is a deterministic function of those draws and the previous
  state. So the kernel is fully specified once you know $P(x), P(y)$ and the
  current urn contents — and you do, because you run the simulator.
- **You do not estimate this kernel; you compute it.** Every numerator and
  denominator is an integer. The transition probability between any two
  specified full states is a rational number in closed form.

The implementation of this is short and worth seeing concretely:

```python
import numpy as np
from itertools import product

def choice_probs(propensity_vec):
    """
    Convert a propensity vector (vector of non-negative integers) into the
    Roth-Erev choice distribution.

    The Roth-Erev choice rule is "probability proportional to propensity":
        P(option k) = n_k / sum(n)
    This is exactly the urn-fraction interpretation of a Polya urn.

    Parameters
    ----------
    propensity_vec : array-like of non-negative numbers
        The current urn contents. Must have a strictly positive total mass; we
        do NOT define behavior for the all-zero urn (the paper avoids this by
        always initializing with at least one ball per option).

    Returns
    -------
    p : np.ndarray of float
        A probability vector summing to 1.0.
    """
    n = np.asarray(propensity_vec, dtype=float)
    total = n.sum()
    if total <= 0:
        raise ValueError("Roth-Erev urn requires strictly positive total mass.")
    return n / total


def one_step_kernel_value(s_curr, s_next, P_x, P_y, G1, G2):
    """
    Compute the EXACT one-step transition probability P(s_next | s_curr) for
    the full Markov chain of the two-agent signal trading game with Roth-Erev
    learning, using the factorization in section 2 of this note.

    The function returns 0 if any of the deterministic constraints are
    violated (rewards inconsistent with the games, or urn updates inconsistent
    with the additive Roth-Erev rule). Otherwise it multiplies the four
    stochastic factors (nature, two signal urns, two action urns).

    Parameters
    ----------
    s_curr, s_next : dict
        States of the chain. Each dict has the keys:
            'x', 'y'       : ints (nature's draws AT this step)
            'f1', 'f2'     : 2D int arrays of shape (|X|, |Sig|)
            'g1', 'g2'     : 3D int arrays of shape (|X|, |Sig|, |Ac|)
                             indexed as g[(obs, sig)][a]
            'sig1', 'sig2' : ints (signals sent at this step)
            'a1', 'a2'     : ints (actions taken at this step)
            'r1', 'r2'     : ints in {0, 1} (rewards at this step)
        s_curr's f, g are the urn states USED in this step (i.e. the state at
        time t before the t->t+1 update). s_next's f, g are the urns AFTER
        update.
    P_x, P_y : array-like
        Marginal distributions of nature's two binary variables.
    G1, G2 : callables
        Reward functions G_i(a, x, y) -> {0, 1}.

    Returns
    -------
    p : float
        The exact transition probability under the factorization.
    """
    x_n, y_n = s_next['x'], s_next['y']
    sig1, sig2 = s_next['sig1'], s_next['sig2']
    a1, a2 = s_next['a1'], s_next['a2']
    r1, r2 = s_next['r1'], s_next['r2']

    # --- Stochastic factor 1: nature -----------------------------------------
    # x and y are sampled independently from their fixed marginal distributions.
    p_nature = P_x[x_n] * P_y[y_n]

    # --- Stochastic factor 2: signal urns ------------------------------------
    # Each agent samples a signal from its f-urn for the OBSERVED nature state.
    # The relevant urns are the rows f1[x_n] and f2[y_n] of the CURRENT state.
    p_sig1 = choice_probs(s_curr['f1'][x_n])[sig1]
    p_sig2 = choice_probs(s_curr['f2'][y_n])[sig2]

    # --- Stochastic factor 3: action urns ------------------------------------
    # Each agent samples an action conditioned on (own observation, signal
    # received from the other agent). Agent 1 receives sig2; agent 2 receives sig1.
    p_a1 = choice_probs(s_curr['g1'][x_n, sig2])[a1]
    p_a2 = choice_probs(s_curr['g2'][y_n, sig1])[a2]

    # --- Deterministic factor 1: rewards must match the matching games -------
    if r1 != G1(a1, x_n, y_n) or r2 != G2(a2, x_n, y_n):
        return 0.0

    # --- Deterministic factor 2: urns must update by the additive rule -------
    # The Roth-Erev update is: increment the cell of the chosen option by r.
    # All other cells are unchanged.
    f1_expected = s_curr['f1'].copy(); f1_expected[x_n, sig1] += r1
    f2_expected = s_curr['f2'].copy(); f2_expected[y_n, sig2] += r2
    g1_expected = s_curr['g1'].copy(); g1_expected[x_n, sig2, a1] += r1
    g2_expected = s_curr['g2'].copy(); g2_expected[y_n, sig1, a2] += r2

    if not (np.array_equal(f1_expected, s_next['f1']) and
            np.array_equal(f2_expected, s_next['f2']) and
            np.array_equal(g1_expected, s_next['g1']) and
            np.array_equal(g2_expected, s_next['g2'])):
        return 0.0

    return p_nature * p_sig1 * p_sig2 * p_a1 * p_a2
```

That function is the complete, exact, no-estimation-needed transition kernel for
the chain defined in §2.3, restricted to Roth–Erev. Everything below this line
is about answering questions you can't ask of the kernel directly — like "what
is the probability of *eventually* ending up near $f^*$?" — for which you do
need either Monte Carlo or a coarse-grained MLE.

---

## 3. The Pólya-urn structure of a single signaling table

Now zoom in on one urn — say agent 1's row $f^{(1)}_t[x]$ for some fixed
nature observation $x$. Let $n = (n_1, \dots, n_K)$ denote its current
contents and $S = \sum_\sigma n_\sigma$. The dynamics of this urn alone, per
episode, are:

- With probability $1 - P(x)$, nature does not draw $x$ and the urn is
  untouched.
- With probability $P(x)$, the urn is consulted: agent 1 samples $\sigma$ with
  probability $n_\sigma / S$, and at the end of the episode the urn is updated
  to $n + r^{(1)} \cdot e_\sigma$ where $e_\sigma$ is the unit vector on
  coordinate $\sigma$.

Let $q_\sigma$ denote the probability that agent 1's reward equals 1 given
that agent 1 sent signal $\sigma$, conditional on $x$ being observed. Then
the per-episode transitions of this single urn are:

$$
\boxed{\;
P\big(n \to n + e_\sigma\big) \;=\; P(x)\cdot \frac{n_\sigma}{S}\cdot q_\sigma,
\qquad
P\big(n \to n\big) \;=\; 1 - \frac{P(x)}{S}\sum_\sigma n_\sigma\, q_\sigma.
\;}
$$

This is the **Pólya-urn-style explicit transition computation** specialized to
Roth–Erev: a draw proportional to current mass, followed by a Bernoulli
reinforcement.

A non-obvious but important observation about the *signaling* urns
specifically:

> **Agent 1's reward $r^{(1)}$ does not depend on the signal $\sigma^{(1)}$
> agent 1 sent.** It depends on $a^{(1)} = g^{(1)}(x, \sigma^{(2)})$, which
> involves the signal received from agent 2, not the one sent to agent 2.
> Therefore $q_\sigma$ for the urn $f^{(1)}[x]$ is the **same for all
> $\sigma$** — call it $q^*(x)$.

So conditional on $x$ being drawn, $f^{(1)}[x]$ is **a pure Pólya urn**:
sample by current proportions, reinforce the chosen color with constant
probability $q^*(x)$ regardless of which color was chosen. By the classical
Pólya theorem (and its straightforward extension to Bernoulli-thinned
reinforcement), the proportion vector

$$
\hat{f}^{(1)}_t[x] \;=\; \frac{f^{(1)}_t[x]}{S^{(1)}_t[x]}
$$

converges almost surely as $t \to \infty$ to a random limit on the simplex,
and the law of that limit is the Dirichlet distribution determined by the
initial propensities. In particular: the signaling table does *not* converge
to a deterministic optimum — it converges to *some* random extreme point on
the simplex, picked out by initial bias and early luck.

This is the formal core of the proof-of-concept: the random selection of *a*
signaling system is delivered by the Pólya structure of the $f$ urns; the
*correctness* of the resulting communication is delivered by the $g$ urns
adapting to whatever the $f$ urns drift into. The two are separately driven
by reinforcement that does have signal-specific selection pressure (because
$g^{(2)}[(y, \sigma)]$ is reinforced when the action it produces succeeds,
and that does depend on $\sigma$).

The complication for a fully rigorous statement is that $q^*(x)$ itself
changes over time as the other agent's $g^{(2)}$ adapts. So the urn is not
autonomous — it's a *generalized* Pólya urn in the sense of Pemantle's
surveys and the stochastic-approximation framework used by Argiento et al.
[Argiento, Pemantle, Skyrms, Volkov, 2009] for the Lewis–Skyrms case. The
single-urn-in-isolation derivation above is the right intuition, but the
joint convergence on the four-urn product simplex is what their theorem
actually proves. The paper rightly flags that extending it to the
signal-trading game is open.

```python
def single_urn_transition_probabilities(n, P_x_obs, q):
    """
    Explicit one-step transition probabilities for a single Roth-Erev urn,
    conditioned on the per-episode reinforcement probabilities `q`.

    The urn is updated only when nature draws the observation x that this urn
    is keyed to (probability P_x_obs). Conditional on that, the agent samples
    color sigma with probability n[sigma] / sum(n), and with probability
    q[sigma] the chosen color gets one extra ball.

    Parameters
    ----------
    n : array-like of non-negative ints
        Current urn contents, length K.
    P_x_obs : float in [0, 1]
        Probability that the nature observation associated with this urn is
        drawn this episode.
    q : array-like of floats in [0, 1]
        Per-color reinforcement probabilities. For the *signal* urn of an
        agent in a 2-agent matching game these are all equal to q*(x)
        (because the agent's own reward doesn't depend on the signal sent).
        For the *action* urn they generally differ across actions.

    Returns
    -------
    next_states : list of np.ndarray
        The list of possible next states, in the same order as transition_probs.
        Includes both the unchanged state n and the K possible n + e_sigma.
    transition_probs : np.ndarray
        Probabilities, summing to 1.
    """
    n = np.asarray(n, dtype=int)
    q = np.asarray(q, dtype=float)
    K = len(n)
    S = n.sum()
    if S <= 0:
        raise ValueError("Urn must have strictly positive total mass.")

    # Probability of moving to n + e_sigma is the product of three factors:
    #   P(this urn is consulted) * P(sigma drawn) * P(reinforcement)
    sample_probs = n / S
    move_probs = P_x_obs * sample_probs * q

    # Probability of staying at n is the residual.
    stay_prob = 1.0 - move_probs.sum()

    # Build the explicit list of (next_state, prob) pairs.
    next_states = [n.copy()]
    probs = [stay_prob]
    for sigma in range(K):
        nn = n.copy()
        nn[sigma] += 1
        next_states.append(nn)
        probs.append(move_probs[sigma])

    return next_states, np.array(probs)


# --- Example -----------------------------------------------------------------
# An agent's signal urn for x=0 with 2 signals, currently in state [3, 1].
# Suppose P(x=0) = 0.5 and q*(0) = 0.6 (success rate when x=0 is observed).
# Note q is independent of sigma for the signal urn — the key Polya-urn fact.
n = [3, 1]
states, probs = single_urn_transition_probabilities(
    n=n, P_x_obs=0.5, q=[0.6, 0.6]
)
for s, p in zip(states, probs):
    print(f"  P({n} -> {list(s)}) = {p:.4f}")
# Expected output:
#   P([3, 1] -> [3, 1]) = 0.7000   = 1 - 0.5 * 0.6 = stay (no consult or no reinforce)
#   P([3, 1] -> [4, 1]) = 0.2250   = 0.5 * 0.75 * 0.6
#   P([3, 1] -> [3, 2]) = 0.0750   = 0.5 * 0.25 * 0.6
```

---

## 4. Why naive transition counting fails on the full state

Take the trajectory $s_0, s_1, \dots, s_T$ from a simulation, define the
empirical transition count $\hat N(s, s') = \#\{t : s_t = s,\ s_{t+1} = s'\}$
and try to row-normalize. Three pathologies appear:

1. Because the chain is non-recurrent, almost every $\hat N(s, s')$ that is
   not zero is exactly 1. The estimator is essentially the indicator that the
   transition was observed, not a probability.
2. The denominator $\sum_{s'} \hat N(s, s')$ is also almost always 1, so
   every row sums to 1 trivially with all mass on the unique successor seen.
3. There is no generalization: the estimated kernel says nothing about
   transitions out of any state you didn't visit, which is almost all of them.

This is the formal content of "MLE is not well-posed on the unreduced state."
The standard Markov-chain MLE — which I described in the previous reply —
does work for *finite* observable chains, and it is consistent and
asymptotically normal there. It just does not apply here.

---

## 5. MLE on a coarse-grained chain

The fix is to project the trajectory onto a discrete feature
$\phi : \mathcal{S} \to \mathcal{Z}$ where $|\mathcal{Z}|$ is small enough
that visits recur. The projected process $\phi(s_t)$ is in general not
exactly Markov, but it is approximately so for many useful $\phi$, and the
plain counting MLE on the projection gives a meaningful matrix.

Three features are useful for §2.3 specifically:

- **Modal signaling map.** For each $x$, take $\arg\max_\sigma f[x, \sigma]$.
  The full $f$-table collapses to a discrete map $X \to \text{Sig}$. With
  $|X| = |\text{Sig}| = 2$ this gives 4 buckets, of which 2 are the ideal
  signaling permutations $f^*$ (and $f^{*-1}$).
- **Normalized-propensity simplex bins.** For each $x$, take $f[x] / S[x]$
  and bin to a coarse grid on the simplex. This carries strictly more
  information than the modal map because it tracks how *concentrated* the
  signaling is, which matters for the Pólya analysis.
- **NMI bins.** Compute the normalized mutual information between $X$ and
  the random variable $\sigma \sim f[x] / S[x]$, and bin. This is the
  metric used in the simulation plots, so estimated transitions on this
  feature directly answer questions like "from NMI bin 0.5, what is the
  empirical probability of ending in bin > 0.9 within $K$ steps?"

```python
import numpy as np
from collections import Counter


def estimate_coarse_transition_matrix(states, feature_fn, smoothing=1.0):
    """
    Plain counting MLE for the transition matrix of a discrete-state Markov
    chain obtained by projecting a continuous- or large-state trajectory
    through a feature map.

    The estimator is:
        A_hat[i, j] = (smoothing + N[i, j]) / (K * smoothing + sum_k N[i, k])
    which is the posterior mean under a symmetric Dirichlet(1 + smoothing)
    prior on each row. With smoothing=0 it reduces to the raw MLE
    A_hat[i, j] = N[i, j] / sum_k N[i, k].

    Caveat. The projected process phi(s_t) is generally not exactly Markov.
    What this function returns is the empirical *one-step* transition kernel
    of the projection, averaged over whatever distribution of within-bucket
    full-states the trajectory happened to visit. That object is still
    useful — it gives empirical reach probabilities between buckets that you
    can compare across initializations — but it should not be reported as
    "the" transition matrix of a coarsened-but-still-Markov chain.

    Parameters
    ----------
    states : sequence
        Trajectory of full states s_0, ..., s_{T-1}. Each entry is whatever
        object your simulator produces.
    feature_fn : callable
        Maps a full state to a hashable discrete label (a tuple of ints, a
        string, etc.). This defines the coarse-graining.
    smoothing : float
        Additive Dirichlet smoothing. Use 0.0 for raw MLE, ~1.0 for a robust
        posterior-mean estimate that does not zero out unobserved transitions.

    Returns
    -------
    A_hat : (K, K) ndarray
        Row-stochastic estimated transition matrix on the coarse states.
    label_to_index : dict
        Maps observed feature labels to row/column indices of A_hat.
    visit_counts : np.ndarray of shape (K,)
        Total visits to each coarse state. Useful for assessing reliability
        of each row — small visit counts mean noisy rows.
    """
    # ----- 1. Project the trajectory once. -----
    labels = [feature_fn(s) for s in states]

    # ----- 2. Build the label vocabulary in order of first appearance. -----
    # Deterministic ordering makes the resulting matrix easier to inspect.
    label_to_index = {}
    for lab in labels:
        if lab not in label_to_index:
            label_to_index[lab] = len(label_to_index)
    K = len(label_to_index)

    # ----- 3. Count consecutive label pairs and per-label visits. -----
    # Initialize counts at `smoothing` to apply additive Dirichlet smoothing
    # uniformly. visit_counts is the unsmoothed count, useful for diagnostics.
    N = np.full((K, K), smoothing, dtype=float)
    visit_counts = np.zeros(K, dtype=int)
    for lab_t, lab_tp1 in zip(labels[:-1], labels[1:]):
        i = label_to_index[lab_t]
        j = label_to_index[lab_tp1]
        N[i, j] += 1.0
        visit_counts[i] += 1
    # The last label still counts as a visit.
    visit_counts[label_to_index[labels[-1]]] += 1

    # ----- 4. Row-normalize. -----
    row_sums = N.sum(axis=1, keepdims=True)
    A_hat = N / row_sums
    return A_hat, label_to_index, visit_counts


# --- Example feature: modal signaling map ------------------------------------
def modal_signaling_map(state):
    """
    Coarse label = (argmax_sigma f1[x=0], argmax_sigma f1[x=1], ...).

    For |X| = |Sig| = 2 this returns one of 4 tuples:
        (0, 0), (0, 1), (1, 0), (1, 1)
    where (0, 1) and (1, 0) are the two ideal signaling permutations f^*.

    Ties are broken by argmax's standard "first index" rule, which is fine
    here because in Roth-Erev with integer propensities exact ties are rare
    after the first few episodes.
    """
    f1 = np.asarray(state['f1'])  # shape (|X|, |Sig|)
    return tuple(int(k) for k in f1.argmax(axis=1))


# --- Example feature: normalized-propensity simplex bin ----------------------
def simplex_bin_factory(n_bins=10):
    """
    Construct a feature function that bins each row of f1 into an n_bins grid
    on the simplex, then concatenates bin tuples across rows.

    This carries more information than the modal map: in particular it
    distinguishes "weakly preferred" from "strongly committed" signaling.
    That matters for the Polya-urn dynamics of section 3, where two states
    with the same modal map but different total mass have very different
    one-step movement probabilities.
    """
    def feature_fn(state):
        f1 = np.asarray(state['f1'], dtype=float)  # shape (|X|, |Sig|)
        # Normalize each row to a probability vector on the simplex.
        row_sums = f1.sum(axis=1, keepdims=True)
        # Avoid /0 in pathological cases; in practice S > 0 always here.
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        p = f1 / row_sums
        # Bin each entry to {0, 1, ..., n_bins-1}. We take min with n_bins-1
        # because p == 1.0 would otherwise map to n_bins (out of range).
        binned = np.minimum((p * n_bins).astype(int), n_bins - 1)
        return tuple(binned.flatten().tolist())
    return feature_fn
```

---

## 6. Validation: does the simulator agree with the exact kernel?

A useful sanity check, and one of the few things worth running unit-test
style for a paper, is to verify that the simulator's *empirical* one-step
behavior matches the *theoretical* one-step kernel from §2. This catches
implementation bugs quickly, and it gives you something concrete to point at
when reviewers ask about correctness.

The check has two layers:

1. **Choice rule.** Fix a state $s$ with urn contents $n$, sample many
   independent next-step *signals* (by repeatedly calling the choice rule
   without updating), and check that empirical frequencies match
   $n / \sum n$ within Monte Carlo error.
2. **Single-urn transition.** Fix a state $n$, simulate full episodes
   without ever updating the *other* agents, count how many times each
   transition $n \to n + e_\sigma$ was realized, and compare to
   $P(x) \cdot (n_\sigma / S) \cdot q^*(x)$ — where $q^*(x)$ is itself
   estimated from the same simulation.

```python
def validate_choice_rule(propensity_vec, n_samples=100_000, rng=None):
    """
    Compare the empirical choice frequencies of the Roth-Erev choice rule to
    the theoretical proportions n / sum(n). Returns the empirical frequencies
    and the maximum absolute deviation from theory; deviation should be
    O(1/sqrt(n_samples)) by the CLT.

    This isolates the choice rule from the urn dynamics: we sample repeatedly
    from the same fixed urn (no update) so the only randomness is the
    sampling step.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    p_theory = choice_probs(propensity_vec)
    # rng.choice is the standard categorical sampler; size=n_samples vectorizes.
    samples = rng.choice(len(p_theory), size=n_samples, p=p_theory)
    # Use np.bincount for the empirical histogram. minlength ensures the
    # output has one entry per option even if some option was never drawn.
    p_empirical = np.bincount(samples, minlength=len(p_theory)) / n_samples
    max_abs_dev = np.max(np.abs(p_empirical - p_theory))
    return p_empirical, p_theory, max_abs_dev
```

The single-urn transition check is best done as a small simulation script
where you instrument the run to count $n \to n + e_\sigma$ events
specifically. The key thing to confirm is the **factorized** form: drawing
$\sigma$ proportional to current mass *and* reinforcing with probability
$q^*(x)$ together produce the boxed formula above. If you compute these two
quantities separately and multiply, the result must agree with the joint
empirical frequency to within $1/\sqrt{T}$.

---

## 7. Summary

For the Roth–Erev signal-trading chain of §2.3:

- **You do not estimate the one-step transition kernel.** You compute it
  exactly by multiplying a small product of urn-fraction terms (function
  `one_step_kernel_value` above). This is what "Pólya-urn-style explicit
  computation" means here: every factor is the urn-fraction $n_\sigma / S$
  times a Bernoulli reinforcement probability, no estimation needed.
- **The signaling urn $f[x]$ is a pure Pólya urn** in the sense that its
  reinforcement probability $q^*(x)$ does not depend on which signal was
  sent. Its proportion vector converges almost surely to a Dirichlet limit
  determined by initial bias. This is the formal content of the "attractor"
  picture in §2.3 and the reason initialization dominates the final
  outcome (Figure 1).
- **The action urns $g$ are not pure Pólya urns** — their reinforcement
  probabilities *do* depend on which action was chosen, so they have
  signal-specific selection pressure. This is what allows agent 2's $g^{(2)}$
  to lock onto whatever signaling system agent 1's drifting $f^{(1)}$ ends up
  in.
- **Naive MLE on the full chain is degenerate** because the chain is
  non-recurrent on its unbounded integer lattice.
- **The right MLE-style estimate is the empirical transition matrix on a
  coarse-grained projection** (modal signaling map, simplex bins, or NMI
  bins), obtained by counting and row-normalizing. Function
  `estimate_coarse_transition_matrix` above is the implementation.
- **The two approaches are complementary.** Use the exact kernel for
  formal claims about specific paths ("the probability that this
  initialization reaches $f^*$ within $K$ steps is at least…"). Use the
  coarse-grained MLE for the empirical transition matrices that show up in
  the proof-of-concept argument and the simulation plots.

---

## References

- Argiento, Pemantle, Skyrms, Volkov (2009). *Learning to signal: Analysis
  of a micro-level reinforcement model.* Stoch. Proc. Appl. — the
  convergence theorem for the Lewis–Skyrms game cited in §2.3.
- Pemantle, R. (2007). *A survey of random processes with reinforcement.*
  Probab. Surveys — standard reference for generalized Pólya urns and
  stochastic-approximation arguments.
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction* (2nd
  ed.) — Robbins–Monro conditions and the convergence of the Q-learning
  averaging used in the paper's stability footnote.
