from imports import *
from utils import *
from agents import UrnAgent
from environment import MultiAgentEnv
from simulation_functions import simulation_function

n_agents = 2
n_features = 2
agents_observed_variables = {0:[0],1:[1]}
n_signaling_actions = 2
n_final_actions = 4

randomcannonical_game = {}
for i in range(n_agents):
  randomcannonical_game[i] = create_randomcannonical_game(n_features,n_final_actions)

with_signals, full_information = True, False


env = MultiAgentEnv(n_agents=n_agents, n_features=n_features,
                  n_signaling_actions=n_signaling_actions,
                  n_final_actions=n_final_actions,
                  full_information = full_information,
                  game_dicts=randomcannonical_game,
                  observed_variables = agents_observed_variables)

signal_usage, rewards_history, signal_information_history, urn_histories,nature_history = simulation_function(n_agents=n_agents,
                      n_features=n_features, n_signaling_actions=n_signaling_actions, n_final_actions=n_final_actions,
                      n_episodes=15, with_signals = with_signals,
                      plot=True,env=env,initialize = False, verbose=True)