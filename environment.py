from imports import *
from utils import *
from agents import UrnAgent, QLearningAgent, QLearningAgentTemporal

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
                initialize=None,initialization_weights = [1,0],
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
                       initialize=initialize,initialization_weights=initialization_weights) for _ in range(self.n_agents)]
        
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
        # Trackings signals and actions
        self.signal_usage = [{} for _ in range(self.n_agents)] # accumulated signal counts for each observation
        self.action_usage = [{} for _ in range(self.n_agents)] # accumulated action counts for each observation
        self.signal_information_history = [[] for _ in range(self.n_agents)]  # Track mutual information history
        self.nature_history = []  # Track nature vector history
        # History of signal and action usage at the end of each episode
        self.histories = {}
        for i, agent in enumerate(self.agents):
            self.histories[i] = {'signal_history':[],'action_history':[]}
      
    def nature_sample(self):
        """
        Reset the environment to its initial state and return the new nature vector.

        :return: Randomly generated binary nature vector.
        """
        self.current_step = 0
        self.nature_vector = np.random.randint(0, 2, size=self.n_features)  # Generate random binary vector
        return self.nature_vector

    def encoding_signals(self, agents_observations):
        """
        Execute the signaling step where agents select their signals on the basis of their observations.
        :param agents_observations: List of observations made by each agent.
        """
        # Assign signals based on agent-specific visibility
        signals = [agent.get_signal(observation) for agent, observation in zip(self.agents, agents_observations)]
        # Update signal usage tracking
        for i in range(self.n_agents):
            agent_observation = agents_observations[i]
            
            # Initialize tracking for this observation if it does not exist
            if agent_observation not in self.signal_usage[i]:
                self.signal_usage[i][agent_observation] = np.zeros(self.n_signaling_actions)
            
            # Ensure valid signal selection
            if not (0 <= signals[i] < self.n_signaling_actions):
                raise ValueError(f"Signal {signals[i]} is out of range for agent {i}")
            else:
                self.signal_usage[i][agent_observation][signals[i]] += 1
        
        # Compute and record mutual information of signals
        for i in range(self.n_agents):
            mutual_info, normalized_mutual_info = compute_mutual_information(self.signal_usage[i])
            self.signal_information_history[i].append(normalized_mutual_info)
        
        # encoding signals is step 2, independently on whether it was sent or not
        self.current_step = 2
        return signals  # Step not yet complete, waiting for final actions

    def send_signals(self, signals,agents_observations):
        """
        Send signals to neighboring agents based on the graph structure.
        Outputs new observations for each agent based on received signals plus old observations.
        """
        new_observations = copy.deepcopy(agents_observations)
        for i, agent in enumerate(self.agents):
            # each agent looks at all the agents that are sending signals to it
          in_neighbors = self.graph.predecessors(i)
          for neig in in_neighbors:
            new_observations[i]=new_observations[i]+(signals[neig],)       
        return new_observations

    def get_actions(self, agents_observations):
        final_actions = [agent.get_action(observation) for agent, observation in zip(self.agents, agents_observations)]
        
        # update action usage tracking
        for i in range(self.n_agents):
            agent_observation = agents_observations[i]
            # Initialize tracking for this observation if it does not exist
            if agent_observation not in self.action_usage[i]:
                self.action_usage[i][agent_observation] = np.zeros(self.n_final_actions)
            
            # Ensure valid action selection
            if not (0 <= final_actions[i] < self.n_final_actions):
                raise ValueError(f"Action {final_actions[i]} is out of range for agent {i}")
            else:
                self.action_usage[i][agent_observation][final_actions[i]] += 1    
            
                
        # get_actions is step 3
        self.current_step = 3
        return final_actions
    
    def play_step(self, final_actions):       
        # get rewards
        # rewards = self.calculate_rewards(final_actions)
        rewards = []
        for i in range(self.n_agents):
            agent_action = final_actions[i]
            state_key = tuple(self.nature_vector)  # Convert nature vector to tuple for dictionary lookup
            
            if state_key in self.internal_game_dicts[i] and agent_action in self.internal_game_dicts[i][state_key]:
                rewards.append(self.internal_game_dicts[i][state_key][agent_action])
            else:
                raise KeyError(f"Invalid state-action pair ({state_key}, {agent_action}) for agent {i}")
        
        # update rewards history
        for i in range(self.n_agents):
            self.rewards_history[i].append(rewards[i])
        
        # play_step is step 4
        self.current_step = 4
        return rewards, True
    
    def update_agents(self, nature_observations,new_observations, signals, final_actions, rewards):
        for i in range(self.n_agents):
            self.agents[i].update_signals(nature_observations[i],signals[i], rewards[i])
            self.agents[i].update_actions(new_observations[i],final_actions[i], rewards[i])
        
        for i, agent in enumerate(self.agents):
            self.histories[i]['signal_history'].append(copy.deepcopy(self.signal_usage[i]))
            self.histories[i]['action_history'].append(copy.deepcopy(self.action_usage[i]))
        # update_agents is step 5
        self.current_step = 5
        
    def report_metrics(self):
        """
        Report key metrics from the environment, including signal usage, rewards history,
        and mutual information history.

        :return: Tuple containing signal usage, rewards history, and signal information history.
        """
        return self.signal_usage, self.rewards_history, self.signal_information_history, self.nature_history, self.histories

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
        self.nature_history.append(tuple(nature_vector))
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
        # Giving observations is step 1
        self.current_step = 1
        return agents_observations


class NetTempMultiAgentEnvTemporal:
    def __init__(self, n_agents=2, n_features=2, n_actions=4,
                 full_information=False, game_dicts=None,
                 observed_variables=None,
                 agent_type=None,
                 graph=None):
        if graph is None:
            raise ValueError("Graph cannot be None.")
        if len(graph.nodes) != n_agents:
            raise ValueError("Number of agents must match number of graph nodes.")

        self.n_agents = n_agents
        self.n_features = n_features
        self.n_actions = n_actions
        self.full_information = full_information
        self.graph = graph
        self.game_dicts = game_dicts or {}
        self.observed_variables = observed_variables or {}

        # Initialize agents
        self.agents = [agent_type(n_actions=n_actions) for _ in range(n_agents)]

        # Internal state
        self.nature_vector = None
        self.rewards_history = [[] for _ in range(n_agents)]
        self.action_usage = [{} for _ in range(n_agents)]
        self.nature_history = []

    def nature_sample(self):
        self.nature_vector = np.random.randint(0, 2, size=self.n_features)
        return self.nature_vector

    def assign_observations(self, nature_vector):
        self.nature_history.append(tuple(nature_vector))
        agents_observations = []
        for i in range(self.n_agents):
            if self.full_information:
                obs = tuple(nature_vector)
            else:
                idxs = self.observed_variables[i]
                obs = tuple(nature_vector[j] for j in idxs)
            agents_observations.append(obs)
        return agents_observations

    def communicate(self, observations):
        """Agents receive messages from their neighbors based on graph."""
        new_obs = list(observations)
        for i in range(self.n_agents):
            for neighbor in self.graph.predecessors(i):
                new_obs[i] = new_obs[i] + (observations[neighbor],)
        return new_obs

    def get_actions(self, observations):
        actions = []
        for i, (agent, obs) in enumerate(zip(self.agents, observations)):
            action = agent.get_action(obs)
            actions.append(action)

            # Track usage
            if obs not in self.action_usage[i]:
                self.action_usage[i][obs] = np.zeros(self.n_actions)
            self.action_usage[i][obs][action] += 1
        return actions

    def play_step(self, actions):
        rewards = []
        for i, action in enumerate(actions):
            state_key = tuple(self.nature_vector)
            reward = self.game_dicts[i].get(state_key, {}).get(action, 0)
            rewards.append(reward)
            self.rewards_history[i].append(reward)
        return rewards, True  # Step done

    def update_agents(self, old_observations, actions, rewards, new_observations, done):
        for i in range(self.n_agents):
            self.agents[i].update(
                state=old_observations[i],
                action=actions[i],
                reward=rewards[i],
                next_state=new_observations[i],
                done=done
            )

    def report_metrics(self):
        return self.action_usage, self.rewards_history, self.nature_history

    def render(self):
        print(f"Nature Vector: {self.nature_vector}")
        print(f"Rewards History: {self.rewards_history}")
        print(f"Action Usage: {self.action_usage}")