from imports import *
from utils import *
from agents import UrnAgent

# Multi-Agent Environment Class
# Environment input
n_agents=2
n_features=2
n_final_actions=4
random_game_dicts = {}
for i in range(n_agents):
  random_game_dicts[i] = create_randomcannonical_game(n_features,n_final_actions)

agents_observed_variables = {0:[0],1:[1]}

class MultiAgentEnv:
    def __init__(self, n_agents=2, n_features=2, n_signaling_actions=2, n_final_actions=4,
                 full_information=False, game_dicts=random_game_dicts,
                 observed_variables=agents_observed_variables):

        self.n_agents = n_agents
        self.n_features = n_features
        self.n_signaling_actions = n_signaling_actions
        self.n_final_actions = n_final_actions
        self.current_step = 0  # Track current step in the environment
        self.full_information = full_information

        # Internal game dictionaries for each agent
        self.internal_game_dicts = game_dicts
        # Observed variables per agent
        self.agents_observed_variables = observed_variables
        # Environment state
        self.nature_vector = None  # Binary vector determined by nature
        self.signals = None  # Signals chosen by agents in step 0
        self.final_actions = None  # Final actions chosen by agents in step 1
        # Tracking history
        self.rewards_history = [[] for _ in range(self.n_agents)]  # Store rewards per episode
        self.signal_usage = [{} for _ in range(self.n_agents)]  # Track signal counts per observation
        self.signal_information_history = [[] for _ in range(self.n_agents)]  # Track mutual information history

    def reset(self):
        self.current_step = 0
        self.nature_vector = np.random.randint(0, 2, size=self.n_features)  # Random binary vector
        self.signals = [None] * self.n_agents  # Reset signals
        self.final_actions = [None] * self.n_agents  # Reset final actions
        return self.nature_vector

    def step(self, actions):
        if self.current_step == 0:
            # Step 0: Agents perform signaling actions
            self.signals = actions
            self.current_step += 1  # Move to the next step
            # Assign observations based on agent-specific visibility
            assigned_observations = self.assign_observations()
            # Update signal usage tracking
            for i in range(self.n_agents):
                agent_observation = assigned_observations[i]
                # Initialize tracking for this observation if it does not exist
                if agent_observation not in self.signal_usage[i]:
                    self.signal_usage[i][agent_observation] = [0] * self.n_signaling_actions
                # Increment signal count for the chosen signal
                self.signal_usage[i][agent_observation][self.signals[i]] += 1
            return False  # Step not yet complete, waiting for final actions

        elif self.current_step == 1:
            # Step 1: Agents perform final actions based on signals
            self.final_actions = actions
            rewards = self.calculate_rewards()  # Compute rewards based on actions
            # Store reward history
            for i in range(self.n_agents):
                self.rewards_history[i].append(rewards[i])
            # Compute and record mutual information of signals
            for i in range(self.n_agents):
                mutual_info, normalized_mutual_info = compute_mutual_information(self.signal_usage[i])
                self.signal_information_history[i].append(normalized_mutual_info)

            return rewards, True  # Step complete, episode ends
        else:
            raise ValueError("Environment has already completed two steps. Reset before reusing.")

    def report_metrics(self):
        return self.signal_usage, self.rewards_history, self.signal_information_history

    def calculate_rewards(self):
        rewards = []
        for i in range(self.n_agents):
            agent_action = self.final_actions[i]

            # Potential issue: Ensure the key exists in the dictionary
            state_key = tuple(self.nature_vector)
            if state_key in self.internal_game_dicts[i]:
                rewards.append(self.internal_game_dicts[i][state_key][agent_action])
            else:
                raise KeyError(f"State {state_key} not found in agent {i}'s game dictionary.")

        return rewards

    def render(self):
        """
        Print the current state of the environment for debugging purposes.
        """
        print(f"Step: {self.current_step}")
        print(f"Nature Vector: {self.nature_vector}")
        print(f"Signals: {self.signals}")
        print(f"Final Actions: {self.final_actions}")

    def assign_observations(self):
        """
        Assign observations to each agent based on their observed variables.

        :return: List of observed feature subsets per agent
        """
        agents_observations = []
        if self.full_information:
            # Each agent sees the full nature vector
            for i in range(self.n_agents):
                agents_observations.append(tuple(self.nature_vector))
        else:
            # Each agent only sees a subset of features
            for i in range(self.n_agents):
                observed_indexes = self.agents_observed_variables[i]
                subset = tuple(self.nature_vector[j] for j in observed_indexes)
                agents_observations.append(subset)

        return agents_observations