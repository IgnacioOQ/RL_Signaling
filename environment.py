from imports import *
from utils import *
from agents import UrnAgent

# Networked Multi-Agent Environment Class
# Environment input
n_agents=2
n_features=2
n_final_actions=4
# example game_dicts and observed variables
random_game_dicts = {}
for i in range(n_agents):
  random_game_dicts[i] = create_random_canonical_game(n_features,n_final_actions)
  
agents_observed_variables = {0:[0],1:[1]}

G = nx.DiGraph()
G.add_nodes_from([0,1])  # Adds multiple nodes at once
G.add_edges_from([(0, 1), (1, 0)])  # Adds multiple edges

class NetMultiAgentEnv:
    def __init__(self, n_agents=2, n_features=2, n_signaling_actions=2, n_final_actions=4,
                full_information=False, game_dicts=None,
                observed_variables=None,
                agent_type=UrnAgent,
                initialize=None,
                graph=None):
        """
        Initialize the multi-agent environment with specified parameters.

        :param n_agents: Number of agents in the environment.
        :param n_features: Number of features in the nature vector.
        :param n_signaling_actions: Number of possible signaling actions.
        :param n_final_actions: Number of possible final actions.
        :param full_information: Boolean indicating if agents have full information.
        :param game_dicts: Dictionary defining the game dynamics for each agent.
        :param observed_variables: Variables observed by each agent.
        :param agent_type: Type of agent to initialize.
        :param initialize: Initialization parameter for agents.
        :param graph: Graph representing agent connectivity.
        """
        if graph is None:
            raise ValueError("Graph cannot be None. Please provide a valid graph structure.")
        
        num_nodes = len(graph.nodes)
        if num_nodes != n_agents:
            raise ValueError(f"Mismatch between number of agents ({n_agents}) and number of nodes in graph ({num_nodes}).")
        
        # Number of agents in the environment
        self.n_agents = n_agents
        
        # Agent type
        self.agent_type = agent_type
        
        # Initialize agents using the specified agent type
        self.agents = [agent_type(n_signaling_actions, n_final_actions,
                       initialize=initialize) for _ in range(self.n_agents)]
        
        # Graph structure representing agent relationships
        self.graph = graph
        
        # Environment parameters
        self.n_features = n_features  # Number of features in the nature vector
        self.n_signaling_actions = n_signaling_actions  # Number of available signaling actions
        self.n_final_actions = n_final_actions  # Number of available final actions
        self.current_step = 0  # Track current step in the environment
        self.full_information = full_information  # Indicates if agents have full knowledge

        # Internal game dictionaries for each agent
        self.internal_game_dicts = game_dicts if game_dicts is not None else {}
        
        # Observed variables per agent
        self.agents_observed_variables = observed_variables if observed_variables is not None else {}
        
        # Environment state variables
        self.nature_vector = None  # Binary vector determined by nature
        self.signals = None  # Signals chosen by agents in step 0
        self.final_actions = None  # Final actions chosen by agents in step 1
        
        # Tracking history of the environment
        self.rewards_history = [[] for _ in range(self.n_agents)]  # Store rewards per episode
        self.signal_usage = [{} for _ in range(self.n_agents)]  # Track signal counts per observation
        self.signal_information_history = [[] for _ in range(self.n_agents)]  # Track mutual information history

    def reset(self):
        """
        Reset the environment to its initial state and return the new nature vector.

        :return: Randomly generated binary nature vector.
        """
        self.current_step = 0
        self.nature_vector = np.random.randint(0, 2, size=self.n_features)  # Generate random binary vector
        return self.nature_vector

    def signals_step(self, signals, nature_vector):
        """
        Execute the signaling step where agents select their signals.

        :param signals: List of signals chosen by the agents.
        :param nature_vector: The current nature vector.
        :return: Boolean indicating if the step is complete.
        """
        # Assign observations based on agent-specific visibility
        assigned_observations = self.assign_observations(nature_vector)
        
        # Update signal usage tracking
        for i in range(self.n_agents):
            agent_observation = assigned_observations[i]
            
            # Initialize tracking for this observation if it does not exist
            if agent_observation not in self.signal_usage[i]:
                self.signal_usage[i][agent_observation] = [0] * self.n_signaling_actions
            
            # Ensure valid signal selection
            if not (0 <= signals[i] < self.n_signaling_actions):
                raise ValueError(f"Signal {signals[i]} is out of range for agent {i}")
            else:
                self.signal_usage[i][agent_observation][signals[i]] += 1
        
        self.current_step = 1
        return False  # Step not yet complete, waiting for final actions

    def actions_step(self, actions):
        """
        Execute the final action step where agents make their decisions.

        :param actions: List of actions performed by the agents.
        :return: Tuple of rewards and completion status.
        """
        # Compute rewards based on actions
        rewards = self.calculate_rewards(actions)
        
        # Store reward history
        for i in range(self.n_agents):
            self.rewards_history[i].append(rewards[i])
        
        # Compute and record mutual information of signals
        for i in range(self.n_agents):
            mutual_info, normalized_mutual_info = compute_mutual_information(self.signal_usage[i])
            self.signal_information_history[i].append(normalized_mutual_info)

        return rewards, True  # Step complete, episode ends

    def report_metrics(self):
        """
        Report key metrics from the environment, including signal usage, rewards history,
        and mutual information history.

        :return: Tuple containing signal usage, rewards history, and signal information history.
        """
        return self.signal_usage, self.rewards_history, self.signal_information_history

    def calculate_rewards(self, actions):
        """
        Calculate the rewards for each agent based on the final actions and nature vector.

        :param actions: List of actions chosen by the agents.
        :return: List of rewards for each agent.
        """
        rewards = []
        for i in range(self.n_agents):
            agent_action = actions[i]
            state_key = tuple(self.nature_vector)  # Convert nature vector to tuple for dictionary lookup
            
            if state_key in self.internal_game_dicts[i] and agent_action in self.internal_game_dicts[i][state_key]:
                rewards.append(self.internal_game_dicts[i][state_key][agent_action])
            else:
                raise KeyError(f"Invalid state-action pair ({state_key}, {agent_action}) for agent {i}")
        
        return rewards

    def render(self):
        """
        Print the current state of the environment for debugging purposes.
        """
        print(f"Step: {self.current_step}")
        print(f"Nature Vector: {self.nature_vector}")
        print(f"Signals: {self.signals}")
        print(f"Final Actions: {self.final_actions}")

    def assign_observations(self, nature_vector):
        """
        Assign observations to each agent based on their observed variables.

        :param nature_vector: The environment's nature vector.
        :return: List of observed feature subsets per agent.
        """
        agents_observations = []
        if self.full_information:
            # Each agent sees the full nature vector
            for _ in range(self.n_agents):
                agents_observations.append(tuple(nature_vector))
        else:
            # Each agent only sees a subset of features
            for i in range(self.n_agents):
                observed_indexes = self.agents_observed_variables[i]
                subset = tuple(nature_vector[j] for j in observed_indexes)
                agents_observations.append(subset)

        return agents_observations
