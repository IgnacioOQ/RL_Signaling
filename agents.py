from imports import *
from utils import *

# Urn-Learning Agent
class UrnAgent:
    def __init__(self, n_signaling_actions, n_final_actions, 
                 # these are dummy parameters for the urn agent, but they help with generalization
                 # and are used in the QLearningAgent
                 # and TDLearningAgent
                 exploration_rate=1.0,
                exploration_decay=0.995, min_exploration_rate=0.001,
                # these are not dummy
                 n_observed_features=1, 
                 initialize=False,initialization_weights = [1,0]):
        """
        Initialize the UrnAgent.

        Parameters:
        - n_signaling_actions (int): Number of possible signaling actions.
        - n_final_actions (int): Number of possible final actions.
        - n_observed_features (int): Number of observed features (default is 1).
        - initialize (bool): Whether to initialize the signaling urns with predefined values.
        """
        self.n_signaling_actions = n_signaling_actions
        self.n_final_actions = n_final_actions

        if initialize:
            self.signaling_urns = create_initial_signals(n_observed_features=n_observed_features,
                                                          n_signals=n_signaling_actions, n=initialization_weights[0], 
                                                          m=initialization_weights[1])
            self.action_urns = create_initial_signals(n_observed_features=n_observed_features+1,
                                                       n_signals=n_final_actions, n=initialization_weights[0], 
                                                       m=initialization_weights[1])
        else:
            self.signaling_urns = {}
        self.action_urns = {}

    def reset_urns(self):
        """Reset the signaling and action urns to empty dictionaries."""
        self.signaling_urns = {}
        self.action_urns = {}

    def get_signal(self, state):
        """
        Select a signaling action based on the probability distribution from the urn.

        Parameters:
        - state: The current state.

        Returns:
        - int: The chosen signaling action.
        # """
        if state not in self.signaling_urns:
            self.signaling_urns[state] = np.ones(self.n_signaling_actions)
        probability_weights = self.signaling_urns[state] / (np.sum(self.signaling_urns[state]))
        return np.random.choice(self.n_signaling_actions, p=probability_weights)

    def get_action(self, state):
        """
        Select a final action based on the probability distribution from the urn.

        Parameters:
        - state: The current state.

        Returns:
        - int: The chosen final action.
        """
        if state not in self.action_urns:
            self.action_urns[state] = np.ones(self.n_final_actions)
        probability_weights = self.action_urns[state] / (np.sum(self.action_urns[state]))
        return np.random.choice(self.n_final_actions, p=probability_weights)

    def update_signals(self, state, signal, reward):
        """
        Update the signaling urn based on the received reward.

        Parameters:
        - state: The current state.
        - signal (int): The signaling action taken.
        - reward (float): The reward received.
        """
        self.signaling_urns[state][signal] += reward

    def update_actions(self, state, action, reward):
        """
        Update the action urn based on the received reward.

        Parameters:
        - state: The current state.
        - action (int): The final action taken.
        - reward (float): The reward received.
        """
        self.action_urns[state][action] += reward


# Q-Learning Agent
class QLearningAgent:
    def __init__(self, n_signaling_actions, n_final_actions,
                 exploration_rate=1, exploration_decay=0.995, 
                 min_exploration_rate=0.001, initialize=False,initialization_weights = [1,0],
                 n_observed_features=1,choice='egreedy',exp_smoothing=False):
        """
        Initialize the QLearningAgent.

        Parameters:
        - n_signaling_actions (int): Number of possible signaling actions.
        - n_final_actions (int): Number of possible final actions.
        - learning_rate (float): Learning rate for Q-learning updates.
        - exploration_rate (float): Initial exploration rate for epsilon-greedy strategy.
        - exploration_decay (float): Decay rate for exploration.
        - min_exploration_rate (float): Minimum exploration rate.
        - initialize (bool): Whether to initialize the Q-tables with predefined values.
        - n_observed_features (int): Number of observed features (default is 1).
        """
        self.n_signaling_actions = n_signaling_actions
        self.n_final_actions = n_final_actions
        self.choice = choice
        self.exp_smoothing = exp_smoothing
        self.signal_exploration_rate = exploration_rate
        self.action_exploration_rate = exploration_rate 
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.signaling_counts = {}
        self.action_counts = {}
        if initialize:
            self.q_table_signaling = create_initial_signals(n_observed_features=n_observed_features,
                                                            n_signals=n_signaling_actions, n=initialization_weights[0], 
                                                            m=initialization_weights[1])
            for state in self.q_table_signaling:
                self.signaling_counts[state] = np.zeros(self.n_signaling_actions)
        else:
            self.q_table_signaling = {}
        self.q_table_action = {}

    def reset(self):
        """Reset the Q-tables for signaling and actions."""
        self.q_table_signaling = {}
        self.q_table_action = {}
        self.signaling_counts = {}
        self.action_counts = {}

    def get_signal(self, state):
        """
        Choose a signaling action using an epsilon-greedy policy.

        Parameters:
        - state: The current state.

        Returns:
        - int: The chosen signaling action.
        """
        if state not in self.q_table_signaling:
            self.q_table_signaling[state] = np.zeros(self.n_signaling_actions)
            self.signaling_counts[state] = np.zeros(self.n_signaling_actions)
        if self.choice == 'egreedy':
            if random.uniform(0, 1) < self.signal_exploration_rate:
                signal = random.randint(0, self.n_signaling_actions - 1)
            else:
                signal = np.argmax(self.q_table_signaling[state])
        elif self.choice == 'softmax':
            q_values = self.q_table_signaling[state]
            tau = self.signal_exploration_rate
            stable_q = q_values - np.max(q_values)  # for numerical stability
            exp_q = np.exp(stable_q / tau)
            probabilities = exp_q / np.sum(exp_q)
            signal = int(np.random.choice(len(q_values), p=probabilities))
        elif self.choice == 'ucb':
            q_values = self.q_table_signaling[state]
            counts = self.signaling_counts[state] + 1e-5  # avoid division by zero
            total_counts = np.sum(self.signaling_counts[state]) + 1
            ucb_bonus = self.signal_exploration_rate * np.sqrt(np.log(total_counts) / counts)
            ucb_scores = q_values + ucb_bonus
            signal = int(np.argmax(ucb_scores))
            
        self.signaling_counts[state][signal] += 1
        return signal
    
    def get_action(self, state):
        """
        Choose a final action using an epsilon-greedy policy.

        Parameters:
        - state: The current state.

        Returns:
        - int: The chosen final action.
        """
        if state not in self.q_table_action:
            self.q_table_action[state] = np.zeros(self.n_final_actions)
            self.action_counts[state] = np.zeros(self.n_final_actions)
        if self.choice == 'egreedy':
            if random.uniform(0, 1) < self.action_exploration_rate:
                action = random.randint(0, self.n_final_actions - 1)
            else:
                action = np.argmax(self.q_table_action[state])
        elif self.choice == 'softmax':
            q_values = self.q_table_action[state]
            tau = self.action_exploration_rate
            stable_q = q_values - np.max(q_values)  # for numerical stability
            exp_q = np.exp(stable_q / tau)
            probabilities = exp_q / np.sum(exp_q)
            action = int(np.random.choice(len(q_values), p=probabilities))
        elif self.choice == 'ucb':
            q_values = self.q_table_action[state]
            counts = self.action_counts[state] + 1e-5  # avoid division by zero
            total_counts = np.sum(self.action_counts[state]) + 1
            ucb_bonus = self.action_exploration_rate * np.sqrt(np.log(total_counts) / counts)
            ucb_scores = q_values + ucb_bonus
            action = int(np.argmax(ucb_scores))
            
        self.action_counts[state][action] += 1
        return action


    def update_signals(self, state, signal, reward):
        """
        Update the Q-table for signaling actions based on the received reward.

        Parameters:
        - state: The current state.
        - signal (int): The signaling action taken.
        - reward (float): The reward received.
        """
        if self.exp_smoothing:
            # Exponential smoothing update
            alpha = 0.1  # or any fixed value in (0, 1)
            self.q_table_signaling[state][signal] = (
                (1 - alpha) * self.q_table_signaling[state][signal] + alpha * reward)
        else:
            td_target = reward
            td_error = td_target - self.q_table_signaling[state][signal]
            # These do satisfy the Robbins-Monro condition (provided exploration has a minimum rate 
            # # and Every state-action pair is visited infinitely often
            # Option 1: self.action_counts[state][action] > 0 because we increased in get action
            # self.q_table_signaling[state][signal] += td_error/self.signaling_counts[state][signal]
            # Option 2: Smoother
            alpha = 1.0 / (1.0 + self.signaling_counts[state][signal])  # avoids div by zero
            self.q_table_signaling[state][signal] += alpha * td_error
            # Option 3
            # self.q_table_signaling[state][signal] += td_error / np.sqrt(self.signaling_counts[state][signal])


        self.signal_exploration_rate = max(self.min_exploration_rate, self.signal_exploration_rate * self.exploration_decay)

    def update_actions(self, state, action, reward):
        """
        Update the Q-table for final actions based on the received reward.

        Parameters:
        - state: The current state.
        - action (int): The final action taken.
        - reward (float): The reward received.
        """
        if self.exp_smoothing:
            # Exponential smoothing update
            alpha = 0.1  # or any fixed value in (0, 1)
            self.q_table_action[state][action] = (
                (1 - alpha) * self.q_table_action[state][action] + alpha * reward)
        else:   
            td_target = reward
            td_error = td_target - self.q_table_action[state][action]
            # These do satisfy the Robbins-Monro condition (provided exploration has a minimum rate 
            # # and Every state-action pair is visited infinitely often
            # Option 1: self.action_counts[state][action] > 0 because we increased in get action
            # self.q_table_action[state][action] += td_error/self.action_counts[state][action]        
            # Option 2: Smoother
            alpha = 1.0 / (1.0 + self.action_counts[state][action])  # avoids div by zero
            self.q_table_action[state][action] += alpha * td_error
        # Option 3
        # self.q_table_signaling[state][signal] += td_error / np.sqrt(self.signaling_counts[state][signal])
        
        

        self.action_exploration_rate = max(self.min_exploration_rate, self.action_exploration_rate * self.exploration_decay)
          

class TDLearningAgent:
    def __init__(self, n_actions, learning_rate=0.1, exploration_rate=1.0,
                 exploration_decay=0.995, min_exploration_rate=0.001, gamma=1, choice='egreedy'):
        # n_actions: Max Number of possible actions max(n_signaling_actions, n_final_actions)
        self.n_actions = n_actions
        self.choice = choice
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.gamma = gamma
        self.q_table = {}
        self.action_counts = {}

    def get_action(self, state, available_actions=None):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
            self.action_counts[state] = np.zeros(self.n_actions)

        if available_actions is None:
            available_actions = list(range(self.n_actions))

        q_values = self.q_table[state]

        if self.choice == 'egreedy':
            if random.random() < self.exploration_rate:
                action = random.choice(available_actions)
            else:
                action = max(available_actions, key=lambda a: q_values[a])

        elif self.choice == 'softmax':
            tau = max(self.exploration_rate, 1e-6)
            logits = np.array([q_values[a] for a in available_actions])
            stable_logits = logits - np.max(logits)
            exp_logits = np.exp(stable_logits / tau)
            probs = exp_logits / np.sum(exp_logits)
            action = np.random.choice(available_actions, p=probs)

        elif self.choice == 'ucb':
            counts = self.action_counts[state] + 1e-5  # avoid division by zero
            total_counts = np.sum(self.action_counts[state]) + 1
            ucb_bonus = self.exploration_rate * np.sqrt(np.log(total_counts) / counts)
            ucb_scores = q_values + ucb_bonus

            # Mask unavailable actions
            masked_scores = np.full_like(ucb_scores, -np.inf)
            masked_scores[available_actions] = ucb_scores[available_actions]

            action = int(np.argmax(masked_scores))

        else:
            raise ValueError(f"Unknown choice strategy: {self.choice}")

        self.action_counts[state][action] += 1
        return action


    def update(self, state, action, reward, next_state, done):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
            self.action_counts[state] = np.zeros(self.n_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_actions)
            self.action_counts[next_state] = np.zeros(self.n_actions)

        td_target = reward
        if not done:
            # Regular update
            td_target += self.gamma*np.max(self.q_table[next_state])
            # UCB update (commented out, can be used if needed)
            # counts = self.action_counts[next_state] + 1e-5  # avoid division by zero
            # total_counts = np.sum(self.action_counts[next_state]) + 1
            # ucb_bonus = self.exploration_rate * np.sqrt(np.log(total_counts) / counts)
            # ucb_scores = self.q_table[next_state] + ucb_bonus
            # td_target += self.gamma * np.max(ucb_scores)

        td_error = td_target - self.q_table[state][action]

        # These do satisfy the Robbins-Monro condition (provided exploration has a minimum rate 
        # # and Every state-action pair is visited infinitely often
        # Option 1: self.action_counts[state][action] > 0 because we increased in get action
        self.q_table[state][action] += td_error/self.action_counts[state][action]
        # Option 2
        # alpha = 1.0 / (1.0 + self.action_counts[state][action])  # avoids div by zero
        # self.q_table[state][action] += alpha * td_error
        # Option 3
        # self.q_table[state][action] += td_error / np.sqrt(self.action_counts[state][action])


        self.exploration_rate = max(self.min_exploration_rate,
                                    self.exploration_rate * self.exploration_decay)
        
        # self.learning_rate = max(self.min_exploration_rate, self.learning_rate * self.exploration_decay)