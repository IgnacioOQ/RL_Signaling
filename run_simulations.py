# Define column names
column_names = ['iteration','n_signaling_actions','n_final_actions', 'full_information','with_signal',
    'Agent_0_NMI', 'Agent_0_avg_reward', 'Agent_1_NMI', 'Agent_1_avg_reward']

# Create an empty DataFrame with the specified column names
results_df = pd.DataFrame(columns=column_names)

n_iterations = 200

# We store the parameter space independently, because sometimes we want to define this process more elaboratively
parameter_space = {}
for iteration in tqdm(range(n_iterations), desc="Processing parameter dictionary"):
  n_agents = 2 # more agents is not minimal
  # 3 features is exactly what we need to take into account all of the relevant conceptual scenarios
  # provided that the information sets are sampled at random
  n_features = 3
  # Furthermore, once we have the number of features, we can put a max to the relevant number of signalling and final actions
  # by 2*n_features, given that these are binary features
  # this corresponds to the fact in the best case scenario there is an signal and/or action for each world state
  # There is also a relevant minimum of 2 signals or actions.
  # So now we only have two dimensions and we can plot in 3D well
  # i.e. x = n signal actions, y = n final actions, z = reward difference, NMI for each agent
  n_signaling_actions = random.randint(2,2*n_features)
  n_final_actions = random.randint(2,2*n_features)
  # our two input parameter are random numbers between zero and one. If you have other parameters, you can randomly sample from a set, interval, etc.
  parameter_space[iteration] = {'n_agents':n_agents,'n_features':n_features, 'n_signaling_actions':n_signaling_actions,
                                'n_final_actions':n_final_actions}

print(parameter_space.keys)
#for iterations in tqdm(range(n_iterations), desc="Processing"):
for iteration in tqdm(range(n_iterations), desc="Processing processing simulations"):
  # get the paramteres from the dictionary
  parameters = parameter_space[iteration]
  n_agents = parameters['n_agents']
  n_features = parameters['n_features']
  n_signaling_actions = parameters['n_signaling_actions']
  n_final_actions = parameters['n_final_actions']

  random_game_dicts = {}
  for i in range(n_agents):
    random_game_dicts[i] = create_random_game(n_features,n_final_actions)


  env = MultiAgentEnv(n_agents=n_agents, n_features=n_features,
                      n_signaling_actions=n_signaling_actions,
                      n_final_actions=n_final_actions,
                      full_information = full_information,
                      game_dict=random_game_dicts)

  with_signals,full_information = False, False
  results = [iteration,n_signaling_actions,n_final_actions,full_information,with_signals]
  signal_usage, rewards_history = simulation_function(n_agents=n_agents, n_features=n_features,
                        n_signaling_actions=n_signaling_actions, n_final_actions=n_final_actions,
                        n_episodes=n_episodes, with_signals = with_signals, full_information = full_information,
                                                    plot=False,env=env)

  mutual_info_0, normalized_mutual_info_0 = compute_mutual_information(signal_usage[0])
  mutual_info_1, normalized_mutual_info_1 = compute_mutual_information(signal_usage[1])
  output = [normalized_mutual_info_0,np.mean(rewards_history[0]),
            normalized_mutual_info_1,np.mean(rewards_history[1])]
  results+=output
  # Adding the list as a row using loc
  results_df.loc[len(results_df)] = results

  with_signals,full_information = True, False
  results = [iteration,n_signaling_actions,n_final_actions,full_information,with_signals]
  signal_usage, rewards_history = simulation_function(n_agents=n_agents, n_features=n_features,
                        n_signaling_actions=n_signaling_actions, n_final_actions=n_final_actions,
                        n_episodes=n_episodes, with_signals = with_signals, full_information = full_information,
                                                    plot=False,env=env)

  mutual_info_0, normalized_mutual_info_0 = compute_mutual_information(signal_usage[0])
  mutual_info_1, normalized_mutual_info_1 = compute_mutual_information(signal_usage[1])
  output = [normalized_mutual_info_0,np.mean(rewards_history[0]),
            normalized_mutual_info_1,np.mean(rewards_history[1])]
  results+=output
  # Adding the list as a row using loc
  results_df.loc[len(results_df)] = results


  with_signals,full_information = False, True
  results = [iteration,n_signaling_actions,n_final_actions,full_information,with_signals]
  signal_usage, rewards_history = simulation_function(n_agents=n_agents, n_features=n_features,
                        n_signaling_actions=n_signaling_actions, n_final_actions=n_final_actions,
                        n_episodes=n_episodes, with_signals = with_signals, full_information = full_information,
                                                    plot=False,env=env)

  mutual_info_0, normalized_mutual_info_0 = compute_mutual_information(signal_usage[0])
  mutual_info_1, normalized_mutual_info_1 = compute_mutual_information(signal_usage[1])
  output = [normalized_mutual_info_0,np.mean(rewards_history[0]),
            normalized_mutual_info_1,np.mean(rewards_history[1])]
  results+=output
  # Adding the list as a row using loc
  results_df.loc[len(results_df)] = results

results_df.to_csv('first_results.csv', index=False)
