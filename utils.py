from imports import *

# Games

# game dictionary takes as input binary tuples of length 4 and outputs a payoff
def create_random_game(n_features=3,n_final_actions=5):
  random_game_dict = dict()
  world_states = set(product([0, 1], repeat=n_features))
  for w in world_states:
    random_game_dict[w] = dict()
    for a in range(n_final_actions):
      random_game_dict[w][a] = random.randint(0, 9)
  return random_game_dict

# Function to generate unique dictionaries with one key having value n=1 and the rest with m=0
def generate_unique_dicts(n_final_actions,n=1,m=0):
    return [
        {i: (n if i == j else m) for i in range(n_final_actions)}
        for j in range(n_final_actions)
    ]
    
# Updated function to create a game dictionary
def create_randomcannonical_game(n_features, n_final_actions,n=1,m=0):
    random_game_dict = dict()
    world_states = list(product([0, 1], repeat=n_features))
    unique_dicts = generate_unique_dicts(n_final_actions,n,m)

    # Ensure we don't exceed the number of available unique dictionaries
    assert len(world_states) <= len(unique_dicts), "Not enough unique dictionaries for the given states"

    # Shuffle the dictionaries to randomly assign them to states
    random.shuffle(unique_dicts)

    for w, unique_dict in zip(world_states, unique_dicts):
        random_game_dict[w] = dict()
        # Add a payoff for each action
        random_game_dict[w]= unique_dict

    return random_game_dict

# Signal Initiation

# Function to generate a one-hot vector
def generate_hot_vectors(n_signals,n=1,m=0):
    return [np.array([n if i == j else m for i in range(n_signals)]) for j in range(n_signals)]

# Updated function to create a game dictionary
def create_initial_signals(n_observed_features, n_signals,n=1,m=0):
    signalling_urns = dict()
    observed_states = list(product([0, 1], repeat=n_observed_features))
    one_hot_vectors = generate_hot_vectors(n_signals,n,m)

    # Ensure we don't exceed the number of available unique vectors
    assert len(observed_states) <= len(one_hot_vectors), "Not enough unique vectors for the given states"

    # Shuffle the vectors to randomly assign them to states
    random.shuffle(one_hot_vectors)

    for o, vector in zip(observed_states, one_hot_vectors):
        signalling_urns[o] = vector

    return signalling_urns

# Information Theory
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