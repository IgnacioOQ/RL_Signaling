def simulation_function(n_agents=n_agents, n_features=n_features,
                        n_signaling_actions=n_signaling_actions, n_final_actions=n_final_actions,
                        n_episodes=n_episodes, with_signals = True, full_information = False,
                        game_dict=game_dict, plot=True,env=env):
# Training Q-Learning Agents in the Environment
    # Initialize environment and agents

    agents = [QLearningAgent(n_signaling_actions, n_final_actions) for _ in range(n_agents)]

    # Tracking metrics
    # rewards_history = [[] for _ in range(n_agents)]  # Store rewards per episode
    # signal_usage = [{} for _ in range(n_agents)]  # Track signal counts for each agent

    for episode in range(n_episodes):
        # Reset the environment for a new episode
        nature_vector = tuple(env.reset())  # Convert nature vector to a tuple for Q-table indexing
        # total_rewards = [0] * n_agents  # Track total rewards for each agent in the episode

        # Pre Step: Assign observations
        # This is a list of tuples, each tuple corresponding to the observation of
        # the agent in that index
        agents_observations = env.assign_observations()

        # Step 0: Agents choose signaling actions based on Q-learning policy
        signals = [agent.get_action(observation, is_signaling=True) for agent, observation in zip(agents, agents_observations)]
        # # Step 0: Agents choose signaling actions based on Q-learning policy
        # signals = [agent.get_action(observation, is_signaling=True) for agent, observation in zip(agents, observations)]
        next_signals, _, _ = env.step(signals)

        # Here we also consider the possibility that the signals are lost so agents cant observe them
        # Step 1: Agents choose final actions based on Q-learning policy
        new_observations = [obs + (signal,) for obs, signal in zip(agents_observations,signals)]
        #new_observations = [obs1 + (signals[1],), obs2 + (signals[0],)]
        if with_signals:
          final_actions = [agent.get_action(new_obs, is_signaling=False) for agent,new_obs in zip(agents,new_observations)]
        else:
          final_actions = [agent.get_action(observation, is_signaling=False) for agent,observation in zip(agents,agents_observations)]

        next_state, rewards, done = env.step(final_actions)

        # Update Q-tables for signaling and final actions
        # update_q_table(self, state, action, reward, next_state, is_signaling=True):
        for i, agent in enumerate(agents):
          if with_signals:
            agent.update_q_table(tuple(agents_observations[i]), signals[i], rewards[i], new_observations[i], is_signaling=True)
          agent.update_q_table(new_observations[i], final_actions[i], rewards[i], tuple(next_state), is_signaling=False)
          #total_rewards[i] += rewards[i]  # Accumulate rewards

        # Decay exploration rate for each agent
        for agent in agents:
            agent.decay_exploration()

        # # Record rewards for this episode
        # for i in range(n_agents):
        #     rewards_history[i].append(total_rewards[i])
    signal_usage, rewards_history = env.report_metrics()
    if plot:
      # Plot results
      plt.figure(figsize=(12, 6))

      # Plot rewards over episodes
      plt.subplot(1, 2, 1)
      for i in range(n_agents):
          smoothed_rewards = [sum(rewards_history[i][j:j+100]) / 100 for j in range(0, n_episodes, 100)]
          plt.plot(range(0, n_episodes, 100), smoothed_rewards, label=f"Agent {i}")
      plt.title("Average Rewards (Smoothed over 100 episodes)")
      plt.xlabel("Episode")
      plt.ylabel("Average Reward")
      plt.legend()

      # Plot signal usage
      plt.subplot(1, 2, 2)
      for i, usage in enumerate(signal_usage):
          for state, counts in usage.items():
              plt.bar(
                  [f"Agent {i} - {state} - Signal {s}" for s in range(n_signaling_actions)],
                  counts,
                  label=f"Agent {i}, State {state}",
                  alpha=0.7
              )
      plt.title("Signal Usage by State")
      plt.ylabel("Frequency")
      plt.xticks(rotation=90)
      plt.legend()

      plt.tight_layout()
      plt.show()

    return signal_usage, rewards_history

def generate_parameters(n_iterations):
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
    return parameter_space
