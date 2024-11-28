# Q-Learning Agent
class QLearningAgent:
    def __init__(self, n_signaling_actions, n_final_actions, learning_rate=0.1,
                 discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        """
        Initialize a Q-learning agent for signaling and final actions.

        :param n_signaling_actions: Number of possible signaling actions
        :param n_final_actions: Number of possible final actions
        :param learning_rate: Learning rate for updating Q-values
        :param discount_factor: Discount factor for future rewards
        :param exploration_rate: Initial exploration rate (epsilon)
        :param exploration_decay: Factor by which exploration rate decays
        :param min_exploration_rate: Minimum exploration rate
        """
        self.n_signaling_actions = n_signaling_actions
        self.n_final_actions = n_final_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        # Q-tables for signaling and final actions
        self.q_table_signaling = {}
        self.q_table_action = {}

    def get_action(self, state, is_signaling=True):
        """
        Choose an action based on the exploration-exploitation trade-off.

        :param state: Current state (tuple)
        :param is_signaling: If True, choose a signaling action; otherwise, choose a final action
        :return: Chosen action (int)
        """
        q_table = self.q_table_signaling if is_signaling else self.q_table_action
        n_actions = self.n_signaling_actions if is_signaling else self.n_final_actions

        if random.uniform(0, 1) < self.exploration_rate:
            # Exploration: choose a random action
            return random.randint(0, n_actions - 1)
        else:
            # Exploitation: choose the action with the highest Q-value
            if state not in q_table:
                # Initialize Q-values for unseen states
                q_table[state] = np.zeros(n_actions)
            return np.argmax(q_table[state])

    def update_q_table(self, state, action, reward, next_state, is_signaling=True):
        """
        Update the Q-value for the given state-action pair.

        :param state: Current state (tuple)
        :param action: Action taken (int)
        :param reward: Reward received (float)
        :param next_state: Next state after the action (tuple)
        :param is_signaling: If True, update the signaling Q-table; otherwise, update the final action Q-table
        """
        if is_signaling:
            # Update signaling Q-table using max Q-value of the ACTION Q-table as future estimate
            # Namely take the future value of the signal by doing argmax over how much it ended up paying after
            # performing the action
            if next_state not in self.q_table_action:
                self.q_table_action[next_state] = np.zeros(self.n_final_actions)
            future_value = np.max(self.q_table_action[next_state])  # Max Q-value from final actions
        else:
            # Update final Q-table (standard Q-learning)
            future_value = 0 if next_state not in self.q_table_action else np.max(self.q_table_action[next_state])

        # Q-table selection
        q_table = self.q_table_signaling if is_signaling else self.q_table_action
        n_actions = self.n_signaling_actions if is_signaling else self.n_final_actions

        if state not in q_table:
            q_table[state] = np.zeros(n_actions)

        # Q-learning update rule
        td_target = reward + self.discount_factor * future_value
        td_error = td_target - q_table[state][action]
        q_table[state][action] += self.learning_rate * td_error

    def decay_exploration(self):
        """
        Decay the exploration rate (epsilon).
        """
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
