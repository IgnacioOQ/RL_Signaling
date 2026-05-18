# Urn Absorbing States — Where They Are, How Many, What They Look Like

- status: active
- type: explanation
- id: rl_signaling.analytics.urn_absorbing_states
- description: Pedagogical companion to `proof_of_concept_markov.md`. Walks slowly through the combinatorial side of the §2.3 Markov chain: what "absorbing" means here, why a state is absorbing iff every urn cell is one-hot, why there are exactly 48 deterministic-bijection policies per agent (2 × 24) and 48² = 2304 jointly, why exactly 4 of those are ideal, why the mean reward across them is exactly 1/4, and how all of this changes under the new asymmetric initialization regime.
- label: [explanation, math, didactic]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-16
<!-- content -->

## Why this document

[`proof_of_concept_markov.md`](proof_of_concept_markov.md) is the formal reference for the §2.3 Markov chain on policy space — it states the propositions, gives the transition kernel, and proves the Pure-Pólya convergence theorem for single signaling rows. This doc is a slower, more example-driven walk through one specific piece of that picture: the *count* of absorbing states (the "2304" number), how the count is derived, what the absorbing states look like, and how to read the structural reward distribution over them. The two docs cover overlapping material, but the emphasis is different: the formal doc tells you the result; this one tells you why the result has the shape it does and how to think about it.

It is written for the philosophical-paper context: §2.3 of [`manuscript/submission/Signaling_Games_with_Distributed_Rewards__Shortened_.pdf`](../../manuscript/submission/Signaling_Games_with_Distributed_Rewards__Shortened_.pdf) refers to absorbing states informally as "deterministic policies the chain can get stuck at." The reader of the philosophy paper does not need any of the math below, but the author needs to *know* it in order to write the §2.3 caveats correctly. This doc is for the author.

## The setup, briefly

The canonical §2.3 setup has 2 agents, 2 binary features of the world state, 2 signals, 4 actions. Each agent observes one feature, exchanges one signal with the other, then picks an action. Each agent has a per-agent matching game $G_i$ that pays 1 for one action per world state and 0 for the other three.

Each agent maintains two **urn tables** of non-negative weights:

- **Signaling urn $f^{(i)}$.** Keyed by the agent's own observation $v \in \{0, 1\}$. The value at key $v$ is a length-2 vector of weights — one per signal. To emit a signal given observation $v$, the agent samples a signal in proportion to those weights.
- **Action urn $g^{(i)}$.** Keyed by $(\text{own observation}, \text{received signal}) \in \{0, 1\}^2$. The value at each key is a length-4 vector of weights — one per action. To pick an action, the agent samples in proportion to those weights.

A *state* of the Markov chain at time $t$ is the four urn tables $(f^{(0)}_t, g^{(0)}_t, f^{(1)}_t, g^{(1)}_t)$. Per-episode randomness (which world state nature drew, which signal was sent, which action was taken, what reward was collected) are *outputs* of the transition, not part of the state.

## What "absorbing" means

A state $\sigma$ is **absorbing** if the dynamics, starting from $\sigma$, returns to the same $\sigma$ with probability 1 — forever. Equivalently: every possible transition leaves the policy unchanged. (The absolute *magnitudes* of urn cells may grow if a cell keeps getting reinforced; what matters is that the policy — the distribution over signals given each observation, and over actions given each key — never changes.)

Absorbing states are the *fixed points* of the dynamics. Whether the chain reliably ends up at one is a separate question (that's the §2.3 "miracle drift" puzzle). Here we are only asking: where are the fixed points, and what do they look like?

## Why absorbing ⇔ every urn cell is one-hot

Call a row of an urn **one-hot** if exactly one cell is positive and the others are zero. Call a state **deterministic** if every row of every urn is one-hot.

**Claim.** A state of the chain is absorbing iff it is deterministic.

**Why one-hot is absorbing.** Roth–Erev (the `UrnAgent` update rule) is

$$
\text{urn}[a] \;\leftarrow\; \max\!\bigl(0,\; \text{urn}[a] + r\bigr),
$$

where $a$ is the cell that was *sampled* and $r \in \{0, 1\}$ is the reward.

Consider a one-hot row, say `urn = [0, 5, 0, 0]`. Sampling is proportional to weight, so the agent samples cell index 1 with probability 1 (the other cells have probability 0). After the episode the row becomes either `[0, 5, 0, 0]` (if reward = 0) or `[0, 6, 0, 0]` (if reward = 1). In both cases the row is still one-hot at the same coordinate. The *policy* implied by this row — "always pick action 1 at this key" — is unchanged.

This generalizes to a deterministic state: every row is one-hot, every sample is forced, every update keeps each row one-hot. So the joint policy is preserved forever.

**Why non-one-hot is not absorbing.** Suppose at least one row has two positive cells, say `urn = [3, 2, 0, 0]` (sampling: 3/5 vs 2/5). With positive probability the next episode (a) keys into this row, (b) samples cell 1 (the smaller one), and (c) gets reward 1. When all three happen, the row becomes `[3, 3, 0, 0]` — a *different* distribution. The policy has changed; the state is not absorbing.

So absorbing states are exactly the deterministic-policy states, and counting absorbing states reduces to counting deterministic policies. Which is a combinatorics problem.

## Counting deterministic policies per agent

There are two urn tables per agent, each independently:

### Signaling urn — 2 deterministic bijections

The signaling urn has 2 rows (one per value of `own_observation`). Each row must be one-hot at one of 2 signals.

- Each row in isolation: 2 choices of which cell is hot.
- Two rows independently: $2 \times 2 = 4$ deterministic functions from $\{0, 1\}$ to $\{0, 1\}$ in principle.

But not all 4 are reachable from this project's initialization. The helper [`create_initial_signals`](../../rl_signaling/games.py#L123) (used by `UrnAgent` when `initialize=True`) generates one-hot vectors and assigns *different* ones to different observations — explicitly:

```python
random.shuffle(one_hot_vectors)
for o, vector in zip(observed_states, one_hot_vectors):
    signalling_urns[o] = vector
```

So the two rows always get *distinct* one-hot patterns: the signaling map is a **bijection** $\{0, 1\} \to \{0, 1\}$. There are exactly $2! = 2$ such bijections (the identity and the swap), and the chain is restricted to these two.

### Action urn — 24 deterministic bijections

The action urn has 4 rows (one per `(own_observation, received_signal)` key, i.e. one per element of $\{0, 1\}^2$). Each row must be one-hot at one of 4 actions.

- Per row: 4 choices.
- Four rows independently: $4^4 = 256$ deterministic functions in principle.

Again `create_initial_signals` enforces a bijection — the 4 keys get 4 *different* one-hot vectors, one each. So the action policy is a bijection from the 4-element key space to the 4-element action space. There are $4! = 24$ such bijections.

### Per-agent total: 48

The signaling and action urns are independent of each other (different keys, different cell counts):

$$
2 \;\times\; 24 \;=\; 48 \text{ deterministic-bijection policies per agent.}
$$

## Joint count: 48 × 48 = 2304

The two agents have completely independent urn tables — they only interact at runtime through the rewards their joint actions generate. So a joint deterministic-bijection policy is a product:

$$
|\Sigma_\text{abs}| \;=\; 48 \;\times\; 48 \;=\; \boxed{2304}.
$$

The script [`enumerate_absorbing_states.py`](scripts/enumerate_absorbing_states.py) iterates over all 2304 profiles and confirms the count (line 130: `check_exact("joint absorbing states", n_joint, 2304)`).

## A subtlety: bijection-only vs all-deterministic

The 2304 count is for **bijection-only** absorbing states — those reachable from `create_initial_signals` initialization. Mathematically, *any* one-hot urn pattern is absorbing under the Roth–Erev update, including:

- **Constant signaling maps**, e.g., $f^{(i)}(0) = f^{(i)}(1) = 0$ (both observations emit the same signal). Then NMI between observation and signal is 0 — no information is transmitted — but the policy is still one-hot per row, and the row patterns are preserved forever.
- **Non-injective action maps**, e.g., $g^{(i)}(v, s) = 2$ for all $(v, s)$ (every key picks the same action).

These exist as fixed points of the dynamics — they are absorbing — but they cannot be *reached* from a bijection-initialized state because the urn updates only preserve patterns, they don't create new ones. Once a chain is in the bijection-only subspace, it stays there.

The mathematically permissive count of one-hot states would be much larger:

$$
\underbrace{2^2}_{\text{signaling fns}} \;\times\; \underbrace{4^4}_{\text{action fns}} \;\times\; \underbrace{(\text{same for agent 1})}_{} \;=\; 4 \cdot 256 \cdot 4 \cdot 256 \;=\; 1{,}048{,}576.
$$

The 2304 count is the relevant one for this project because that's the slice of absorbing space the bijection initialization can reach. If you ever change the initialization to allow non-bijection one-hot maps (no current code does, but it's a design choice that could be made), the count would balloon.

## Within the 2304: how many are ideal?

Call a joint policy **ideal** if both agents earn mean reward 1 over the four world states. There are exactly **4** ideal states in the 2304, and the count comes out of a clean structural argument.

**Setup.** Agent 0's reward on world state $(v_1, v_2)$ is

$$
G_0(v_1, v_2)\Bigl[\, g^{(0)}\!\bigl(v_1,\; f^{(1)}(v_2)\bigr) \Bigr].
$$

For agent 0 to earn mean reward 1, we need $g^{(0)}\bigl(v_1, f^{(1)}(v_2)\bigr) = \alpha_0^\star(v_1, v_2)$ for every $(v_1, v_2)$, where $\alpha_0^\star(v_1, v_2)$ is the unique action that pays 1 for agent 0 at $(v_1, v_2)$ (the matching game guarantees uniqueness).

**The structural point.** Fix agent 1's signaling bijection $f^{(1)}$. Then for each value of $(v_1, v_2)$, the key into agent 0's action urn is $(v_1, f^{(1)}(v_2))$ — and as $(v_1, v_2)$ ranges over the 4 world states, the key $(v_1, f^{(1)}(v_2))$ ranges over all 4 elements of $\{0, 1\}^2$ exactly once (because $f^{(1)}$ is a bijection). So the perfect-reward constraint on $g^{(0)}$ is **one equation per key**, and there are 4 keys. Since $g^{(0)}$ has 4 cells per key and we're forcing each cell to one specific action, the constraint **uniquely determines $g^{(0)}$ given $f^{(1)}$**.

So:

- 2 choices of $f^{(1)}$ (bijections) → 2 unique compatible $g^{(0)}$ → **2 perfect $(f^{(1)}, g^{(0)})$ pairs**.

By symmetry, there are 2 perfect $(f^{(0)}, g^{(1)})$ pairs. The two halves of the joint policy are independent, so:

$$
|\text{ideal joint states}| \;=\; 2 \times 2 \;=\; 4.
$$

Out of 2304, only 4 are ideal — a fraction of $4/2304 \approx 0.17\%$.

## Within the 2304: the mean reward is exactly 1/4

The mean per-agent reward over the 2304 absorbing states is $1/N_\text{act} = 1/4 = 0.25$.

**Why.** Fix agent 0's signaling bijection $f^{(0)}$ and agent 1's signaling bijection $f^{(1)}$. Average agent 0's reward across all action bijection pairs $(g^{(0)}, g^{(1)})$ uniformly — there are $24 \times 24 = 576$ such pairs.

For a fixed key $(v_1, s) \in \{0, 1\}^2$, ask: as $g^{(0)}$ ranges over all 24 action bijections, how often does $g^{(0)}(v_1, s) = a$ for each $a \in \{0, 1, 2, 3\}$? By symmetry of permutations: $24/4 = 6$ times each. So averaging over $g^{(0)}$ acts like **uniform-random action selection at every key**.

Uniform-random action gives expected reward $1/4$ (one of the 4 actions is correct under the matching game). So the mean of agent 0's reward across the 2304 profiles is exactly $1/4 = 0.25$, independent of the game seed (as long as the game has exactly one reward-1 action per world state, which the canonical matching games do).

The empirical check in [`scripts/enumerate_absorbing_states.py`](scripts/enumerate_absorbing_states.py) §7 confirms this to machine precision: `mean r0 = 0.25` exactly.

## The full marginal distribution

Beyond "4 ideal" and "mean 0.25," the full per-agent marginal distribution over the 2304 profiles has a clean shape (game seed 0, but the shape is invariant across seeds):

| Mean reward (per agent) | Count | Fraction |
|---:|---:|---:|
| 0.00 | 864 | 37.5% |
| 0.25 | 768 | 33.3% |
| 0.50 | 576 | 25.0% |
| 0.75 | 0   | 0.0% |
| 1.00 | 96  | 4.2% |

Two surprises worth noting:

- **No 0.75.** Reward 0.75 means the agent's action is correct on exactly 3 of the 4 world states. Under the matching-game structure, that turns out to be combinatorially impossible — a deterministic action policy that's right on 3 of 4 world states would have to "agree with the optimal map" in a way that the bijection-only constraint forbids. (The full proof is mechanical: enumerate all 96 perfect-reward action maps, perturb one cell, check that the resulting policy can only go to reward 0.5 or 0.25 — there's no path through 0.75.)
- **96 perfect, but only 4 *joint* ideal.** Each agent considered separately has 96 perfect-reward profiles. The joint ideal count is $4$, not $96 \times 96 = 9216$, because most "agent 0 perfect" profiles are incompatible with any "agent 1 perfect" profile under shared signaling bijections.

The joint $(r_0, r_1)$ distribution is what Option C in the [proof-of-concept figures notebook](../../notebooks/proof_of_concept_figures.ipynb) plots as a heatmap.

## Why the old `(1, 0)` regime gave reward 0.25

The empirical observation behind the *old* §2.3 "(1,0) paradox": under symmetric `init_weights = (1, 0)`, the agents reach NMI ≈ 1.0 but mean reward ≈ 0.25 — i.e., perfect informational transmission but random-action-baseline reward.

The structural picture above explains this exactly. Under symmetric $(1, 0)$ initialization:

- Both urns are one-hot bijections from $t = 0$.
- The chain is at an absorbing state (forever).
- The starting state is sampled *uniformly* over the 2304 bijection-only absorbing states, because `create_initial_signals` calls `random.shuffle` independently for each of the four channels.
- So the *distribution* of final rewards is exactly the marginal distribution over the 2304 — which has mean $1/4$.

NMI = 1.0 because signals are deterministic functions of observations; reward = 0.25 because the absorbing space is mostly low-reward and the chain is locked at a uniformly random member of it. This is the cleanest *negative* result the §2.3 framework can state: under no learning, you sit where you start, and where you start is bad on average.

## Under the new `sig=[1,0], act=[1,1]` regime

The 2026-05-16 refactor moved the philosophy paper's `(1, 0)` regime from symmetric to **asymmetric**: signaling urns one-hot, action urns uniform. The structural picture above no longer maps cleanly to the dynamics:

- **Signaling urns** are still one-hot bijections from $t = 0$. They are absorbing as before. 2 bijections per agent, 4 joint signaling configurations.
- **Action urns** start at `[1, 1, 1, 1]` per row. They are *never exactly one-hot in finite time* — at each step the sampled cell gets incremented but the others never go to 0 because they started at 1, not 0. Per the Pure-Pólya theorem ([`proof_of_concept_markov.md`](proof_of_concept_markov.md) §"Pure-Pólya signaling-urn convergence"), the row's proportions converge almost surely to a Dirichlet limit, but the limit is generically not at a vertex.

So under the new init the joint chain **does not visit any absorbing state in the strict sense**. It instead concentrates *in distribution* near a one-hot action policy whose specific identity is shaped by which (frozen) signaling bijection got picked at $t = 0$ and which actions paid off as the action urn evolved.

This means the 2304 structural picture is no longer the direct skeleton of the (1, 0) trajectory in Figure 1. It is still informative as a *background* count — "of the bijection-only deterministic policies the dynamics could in principle approach, only 4/2304 are ideal" — but it is not "where the chain ends up." The chain's actual endpoint is some random measure supported near a small subset of one-hot action policies, which is a richer object than a single absorbing state.

For §2.3 this means Option C (the absorbing-state distribution figure) is still a useful structural statement — about the geometry of the deterministic-policy subspace and the bottom-heaviness of distributed-reward absorbing states — but its caption should not claim it explains the (new) Figure 1 blue line's behavior. The relationship is now: blue's signaling lives in the 4-element bijection space; blue's actions don't live in the 24-element bijection space, they live in the simplex.

## Pointers

| Topic | Where |
|---|---|
| Formal absorbing-state propositions and proofs | [`proof_of_concept_markov.md`](proof_of_concept_markov.md) §"Absorbing states under `init_weights = (n, 0)`" |
| Transition kernel for the Markov chain | [`proof_of_concept_markov.md`](proof_of_concept_markov.md) §"Transition kernel" |
| Pure-Pólya theorem for a single signaling row | [`proof_of_concept_markov.md`](proof_of_concept_markov.md) §"Pure-Pólya signaling-urn convergence" |
| Empirical enumeration of all 2304 profiles | [`scripts/enumerate_absorbing_states.py`](scripts/enumerate_absorbing_states.py) |
| Figure of the joint reward distribution | [`scripts/figure_poc_options.py`](scripts/figure_poc_options.py) Option C |
| §2.3 paper-draft connection | [`Proof of Concept (Paper Draft).md`](Proof%20of%20Concept%20(Paper%20Draft).md) |
