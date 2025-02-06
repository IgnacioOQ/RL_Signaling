from imports import *
from utils import *

# Urn-Learning Agent
class UrnAgent:
    def __init__(self, n_signaling_actions, n_final_actions, n_observed_features=1, initialize=False):
        self.n_signaling_actions = n_signaling_actions
        self.n_final_actions = n_final_actions

        if initialize:
            self.signalling_urns = create_initial_signals(n_observed_features=n_observed_features,
                                                          n_signals=n_signaling_actions, n=1, m=0)
        else:
            self.signalling_urns = {}
        self.action_urns = {}

    def reset_urns(self):
        self.signalling_urns = {}
        self.action_urns = {}

    def get_action(self, state, is_signaling=True):
        urn_dict = self.signalling_urns if is_signaling else self.action_urns
        n_actions = self.n_signaling_actions if is_signaling else self.n_final_actions

        if state not in urn_dict:
            urn_dict[state] = np.ones(n_actions)

        epsilon = 1e-8  # Prevent division by zero
        probability_weights = urn_dict[state] / (np.sum(urn_dict[state]) + epsilon)
        return np.random.choice(n_actions, p=probability_weights)

    def update(self, state, action, reward, is_signaling=True):
        urn_dict = self.signalling_urns if is_signaling else self.action_urns
        n_actions = self.n_signaling_actions if is_signaling else self.n_final_actions

        if state not in urn_dict:
            urn_dict[state] = np.ones(n_actions)

        urn_dict[state][action] += reward  # Assuming reward can be float

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, n_signaling_actions, n_final_actions, learning_rate=0.05,
                 exploration_rate=1.0, exploration_decay=0.995, 
                 min_exploration_rate=0.001, initialize=False,
                n_observed_features=1):

        self.n_signaling_actions = n_signaling_actions
        self.n_final_actions = n_final_actions
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        # Q-tables for signaling and final actions
        if initialize:
            self.q_table_signaling = create_initial_signals(n_observed_features=n_observed_features,
                                                n_signals=n_signaling_actions, n=100, m=0)
        else:
            self.q_table_signaling = {}
        self.q_table_action = {}

    def get_action(self, state, is_signaling=True):
        n_actions = self.n_signaling_actions if is_signaling else self.n_final_actions

        if is_signaling:
          if state not in self.q_table_signaling:
              self.q_table_signaling[state] = np.zeros(n_actions)
        else:
          if state not in self.q_table_action:
              self.q_table_action[state] = np.zeros(n_actions)

        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, n_actions - 1)
        else:      # Exploitation: choose the action with the highest Q-value
          if is_signaling:
            return np.argmax(self.q_table_signaling[state])
          else:
            return np.argmax(self.q_table_action[state])

    def update(self, state, action, reward, is_signaling=True):
        n_actions = self.n_signaling_actions if is_signaling else self.n_final_actions

        if is_signaling:
          if state not in self.q_table_signaling:
              self.q_table_signaling[state] = np.zeros(n_actions)
        else:
          if state not in self.q_table_action:
              self.q_table_action[state] = np.zeros(n_actions)

        # Q-learning update rule
        td_target = reward #+ self.discount_factor * future_value
        if is_signaling:
            td_error = td_target - self.q_table_signaling[state][action]
            self.q_table_signaling[state][action]+= self.learning_rate * td_error
        else:
            td_error = td_target - self.q_table_action[state][action]
            self.q_table_action[state][action] += self.learning_rate * td_error
            # Decay exploration rate
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)     
