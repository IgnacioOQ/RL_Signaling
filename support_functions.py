
# game dictionary takes as input binary tuples of length 4 and outputs a payoff
def create_random_game(n_final_actions,n_features):
  random_game_dict = dict()
  world_states = set(product([0, 1], repeat=n_features))
  for w in world_states:
    random_game_dict[w] = dict()
    for a in range(n_final_actions):
      random_game_dict[w][a] = random.randint(0, 9)
  return random_game_dict


def compute_entropy(probabilities):
    """
    Compute entropy from a probability distribution.
    :param probabilities: List of probabilities.
    :return: Entropy value.
    """
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

def compute_mutual_information(agent_signal_usage):
    """
    Compute mutual information and normalized mutual information between signals and observations.
    :param agent_signal_usage: Dictionary tracking signal counts per observation. This is for a single agent.
    :return: Mutual information (MI) and Normalized Mutual Information (NMI).
    """
    # Flatten signal counts to compute overall probabilities
    total_signals = sum(sum(counts) for counts in agent_signal_usage.values())

    # Compute P(S): Overall signal probabilities
    signal_counts = defaultdict(int)
    for counts in agent_signal_usage.values():
        for s, count in enumerate(counts):
            signal_counts[s] += count
    P_S = {s: count / total_signals for s, count in signal_counts.items()}

    # Compute P(O): Observation probabilities
    P_O = {o: sum(counts) / total_signals for o, counts in agent_signal_usage.items()}

    # Compute H(S): Entropy of signals
    H_S = compute_entropy(P_S.values())

    # Compute H(S | O): Conditional entropy of signals given observations
    H_S_given_O = 0
    for o, counts in agent_signal_usage.items():
        P_S_given_O = [count / sum(counts) for count in counts]
        H_S_given_O += P_O[o] * compute_entropy(P_S_given_O)

    # Compute H(O): Entropy of observations
    H_O = compute_entropy(P_O.values())

    # Mutual Information: I(S; O) = H(S) - H(S | O)
    I_S_O = H_S - H_S_given_O

    # Normalized Mutual Information: NMI = I(S; O) / H(O)
    NMI = I_S_O / H_O if H_O > 0 else 0

    return I_S_O, NMI

# Example usage after training
for i, agent_signal_usage in enumerate(signal_usage):
    mutual_info, normalized_mutual_info = compute_mutual_information(agent_signal_usage)
    print(f"Agent {i}:")
    print(f"  Mutual Information (MI): {mutual_info:.4f}")
    print(f"  Normalized Mutual Information (NMI): {normalized_mutual_info:.4f}")
