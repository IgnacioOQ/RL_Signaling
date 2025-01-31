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

# print(create_initial_signals(n_observed_features=2, n_signals=4,n=1,m=0))

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


# PLOTTING
def plot_hist(df,variablewith_signal=True,full_information=False):
  subset_df = df[(df['full_information'] == full_information) & (df['with_signal'] == with_signal)]
  subset_df[variable].plot(kind='hist', bins=100, title=f'variable={variable}, setup = {with_signal,full_information}')
  plt.gca().spines[['top', 'right',]].set_visible(False)
  plt.show()
  
def plot_histograms_with_kde(df, variable, bins=100, figsize=(6, 3),
                             alpha=0.5, kde=True,variables = [(True, True), (True, False), (False, True), (False, False)]):
    # Initialize the figure
    plt.figure(figsize=figsize)

    # Define colors for different setups
    colors = ['blue', 'orange', 'green', 'red']

    # Loop over the combinations of conditions
    for idx, (with_signals, full_information) in enumerate(variables):
        subset_df = df[(df['full_information'] == full_information) & (df['with_signals'] == with_signals)]

        # Plot the histogram for each subset
        plt.hist(
            subset_df[variable],
            bins=bins,
            alpha=alpha,
            color=colors[idx],
            label=f'signals={with_signals}, full_info={full_information}',
            density=True  # Normalize histogram to show density
        )

        # Plot KDE curve if enabled
        if kde:
            sns.kdeplot(
                subset_df[variable],
                color=colors[idx],
                linewidth=2#,
                #label=f'KDE: signals={with_signals}, full_infon={full_information}'
            )

       # Add a vertical line for the mean
        mean_value = subset_df[variable].mean()
        std_dev = subset_df[variable].std()

        plt.axvline(mean_value, color=colors[idx], linestyle='--', linewidth=1.5, 
            label=f'Mean: {mean_value:.2f}, Std Dev: {std_dev:.2f}')
       #plt.axvline(mean_value, color=colors[idx], linestyle='--', linewidth=1.5)#,
                    #label=f'Mean: signals={with_signals}, full_info={full_information}')

        # Initialize an offset value to adjust text positions
        vertical_offset = 0.07  # Adjust as needed to prevent overlap

        # Annotate the mean value on the plot
        plt.text(
            mean_value,
            plt.gca().get_ylim()[1] * (0.9 - idx * vertical_offset),  # Incrementally adjust position
            f'{mean_value:.2f}',
            color=colors[idx],
            fontsize=10,
            ha='center',
            bbox=dict(facecolor='white', edgecolor=colors[idx], boxstyle='round,pad=0.3'))

    # Add labels, legend, and title
    plt.title(f'Histogram and KDE of {variable} by Setup', fontsize=12)
    plt.xlabel(variable, fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.legend(title="Setup", fontsize=9, title_fontsize=10)
    plt.gca().spines[['top', 'right']].set_visible(False)

    # Display the plot
    plt.show()

def plot_all_histograms(df,bins=75):
    plot_histograms_with_kde(df,'Agent_0_final_reward',bins=75)
    plot_histograms_with_kde(df,'Agent_0_avg_reward',bins=75)
    plot_histograms_with_kde(df,'Agent_1_final_reward',bins=75)
    plot_histograms_with_kde(df,'Agent_1_avg_reward',bins=75)
    plot_histograms_with_kde(df,'Agent_0_NMI',bins=75)
    plot_histograms_with_kde(df,'Agent_1_NMI',bins=75)
    df['Agent_0_NMI_Difference'] = df['Agent_0_NMI'] - df['Agent_0_Initial_NMI']
    df['Agent_1_NMI_Difference'] = df['Agent_1_NMI'] - df['Agent_1_Initial_NMI']
    plot_histograms_with_kde(df,'Agent_0_NMI_Difference',bins=50, variables = [(True, True), (True, False)])
    plot_histograms_with_kde(df,'Agent_1_NMI_Difference',bins=50, variables = [(True, True), (True, False)])
    
# Helper function to calculate reward differences
def calculate_reward_difference(df, agent_col):
    return (
        df[df['with_signals']][agent_col].values -
        df[~df['with_signals']][agent_col].values
    )[0]  # Extract the single value

def compare_payoffs(df):
    iteration_indexes = df['iteration'].unique()

    # Define the structure of the resulting DataFrame
    columns = [
        'iteration', 'n_signaling_actions', 'n_final_actions',
        'A0_final_reward_signalvsnon_partialinfo', 'A0_final_reward_signalvsnon_fullinfo',
        'A1_final_reward_signalvsnon_partialinfo', 'A1_final_reward_signalvsnon_fullinfo'
    ]
    compared_payoff_df = pd.DataFrame(columns=columns)

    # Iterate over unique iterations
    for i in iteration_indexes:
        iteration_df = df[df['iteration'] == i]
        n_signaling_actions = iteration_df['n_signaling_actions'].iloc[0]
        n_final_actions = iteration_df['n_final_actions'].iloc[0]

        # Split the DataFrame based on information availability
        full_info = iteration_df[iteration_df['full_information']]
        partial_info = iteration_df[~iteration_df['full_information']]

        # Calculate reward differences
        A0_fullinfo_diff = calculate_reward_difference(full_info, 'Agent_0_final_reward')
        A0_partialinfo_diff = calculate_reward_difference(partial_info, 'Agent_0_final_reward')
        A1_fullinfo_diff = calculate_reward_difference(full_info, 'Agent_1_final_reward')
        A1_partialinfo_diff = calculate_reward_difference(partial_info, 'Agent_1_final_reward')

        # Append results to the DataFrame
        compared_payoff_df.loc[len(compared_payoff_df)] = [
            i, n_signaling_actions, n_final_actions,
            A0_partialinfo_diff, A0_fullinfo_diff,
            A1_partialinfo_diff, A1_fullinfo_diff
        ]
    return compared_payoff_df

def plot_payoff_comparison(df):
    compared_payoff_df = compare_payoffs(df)

    # Define variables
    variables = [
        'A0_final_reward_signalvsnon_partialinfo', 'A0_final_reward_signalvsnon_fullinfo',
        'A1_final_reward_signalvsnon_partialinfo', 'A1_final_reward_signalvsnon_fullinfo'
    ]

    # Define distinct colors for the mean lines
    mean_colors = ['red', 'blue', 'green', 'purple']

    # Plot all variables in the same plot
    plt.figure(figsize=(10, 6))  # Set figure size
    for idx, variable in enumerate(variables):
        # Plot the histogram
        compared_payoff_df[variable].plot(kind='hist', bins=50, alpha=0.6, label=variable)

        # Calculate and plot the mean with a distinct color
        mean_value = compared_payoff_df[variable].mean()
        plt.axvline(mean_value, color=mean_colors[idx], linestyle='--', linewidth=1.5, label=f'{variable} Mean: {mean_value:.2f}')

    # Style the plot
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.title("Distributions of Difference Signaling vs Not")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()  # Add legend to differentiate variables and their means
    plt.show()