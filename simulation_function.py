from imports import *
from utils import *
from agents import UrnAgent, QLearningAgent, TDLearningAgent
from environment import NetMultiAgentEnv, TempNetMultiAgentEnv

n_agents = 2
n_features = 2
n_signaling_actions = 2
n_final_actions = 4

random_game_dicts = {}
for i in range(n_agents):
  random_game_dicts[i] = create_random_canonical_game(n_features,n_final_actions)

agents_observed_variables = {0:[0],1:[1]}

def simulation_function(n_agents=n_agents, n_features=n_features,
                        n_signaling_actions=n_signaling_actions, n_final_actions=n_final_actions,
                        n_episodes=6000, with_signals = True,
                        plot=True,env=None,
                        verbose=False):

    # agents = [agent_type(n_signaling_actions, n_final_actions,
    #                    initialize=initialize) for _ in range(n_agents)]

    # History of the information status of the agent after each episode
    # namely their signalling and action urns as they get more complex

    for episode in range(n_episodes):
      if verbose:
        print(f'episode number is {episode}')
        
      # Step 1: Nature samples state at random and we assign observations to agents
      # Reset the environment for a new episode
      nature_vector = tuple(env.nature_sample())  # Convert nature vector to a tuple for Q-table indexing
      # list of observations for each agent, and updates environment nature history
      agents_observations = env.assign_observations(nature_vector)
      if verbose:
        print(f'nature vector is {nature_vector}')
        print(f'agents direct observations are {agents_observations}')
        print(f'environment step is {env.current_step}')

      # Step 2: Agents choose signaling actions based on learning policy, and 
      # step to store signaling history, and move to the next step in the episode
      signals = env.encoding_signals(agents_observations)
      if with_signals:
        # here I use the environment graph
        # crucial is that the index of the agent corresponds to the name of the node in the graph corresponding to that agent
        new_observations = env.send_signals(signals,agents_observations)
        # print(f'new observations are {new_observations}')
      else:
        new_observations = copy.deepcopy(agents_observations)
      if verbose:
        print(f'agents signals are {signals}')
        print(f'agents new_observations are {new_observations}')
        print(f'environment step is {env.current_step}')
        
      # Step 3: Agents choose final actions based new observations
      final_actions = env.get_actions(new_observations)
      if verbose:
        print(f'agents final_actions are {final_actions}')
        print(f'environment step is {env.current_step}')
        
      # Step 4: Agents receive rewards
      rewards, done = env.play_step(final_actions)
      
      # Step 5: Update agents' signaling and action urns and histories
      env.update_agents(agents_observations,new_observations,signals, final_actions, rewards)
      if verbose:
        print(f'agents rewards are {rewards}')
        print(f'environment step is {env.current_step}')
        
    signal_usage, rewards_history, signal_information_history, nature_history, histories = env.report_metrics()

    if plot:
      # Plot rewards over episodes
      plt.figure(figsize=(8, 5)) # (width, height)
      for i in range(n_agents):
        # first smoothing
        #smoothed_rewards = [sum(rewards_history[i][j:j+100]) / 100 for j in range(0, n_episodes, 100)]
        #plt.plot(range(0, n_episodes, 100), smoothed_rewards, label=f"Agent {i}")
        # second smoothing
        window_size = 100
        smoothed_rewards = np.convolve(rewards_history[i], np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, n_episodes), smoothed_rewards, label=f"Agent {i}")


      plt.title("Average Rewards (Smoothed)")
      plt.xlabel("Episode")
      plt.ylabel("Average Reward")
      plt.legend()

      # Plot NMI over episodes
      plt.figure(figsize=(8, 5)) # (width, height)
      for i in range(n_agents):
        smoothed_NMI = [sum(signal_information_history[i][j:j+10]) / 10 for j in range(0, n_episodes, 10)]
        plt.plot(range(0, n_episodes, 10), smoothed_NMI, label=f"Agent {i}")
      plt.title("Average Normalized Mutual Information (Smoothed)")
      plt.xlabel("Episode")
      plt.ylabel("Average NMI")
      plt.legend()

      # Plot total signal usage
      plt.figure(figsize=(8, 5)) # (width, height)
      for i, usage in enumerate(signal_usage):
          for state, counts in usage.items():
              bar_labels = [f"{count:.2f}" for count in counts]  # Format proportion labels
              bars = plt.bar(
                  [f"A{i}-{state}-Sig {s}" for s in range(n_signaling_actions)],
                  counts,
                  label=f"A{i}, State {state}",
                  alpha=0.7
              )

              # Add proportion labels on top of each bar
              for bar, label in zip(bars, bar_labels):
                  plt.text(
                      bar.get_x() + bar.get_width() / 2,  # Center horizontally
                      bar.get_height(),                   # Position at the top of the bar
                      label,                              # The proportion label
                      ha='center',                        # Horizontal alignment
                      va='bottom'                         # Vertical alignment
                  )
              # plt.bar(
              #     [f"A{i}-{state}-Sig {s}" for s in range(n_signaling_actions)],
              #     counts,
              #     label=f"A{i}, State {state}",
              #     alpha=0.7
              # )
      plt.title("Accumulated Signal Usage Count by Observation")
      plt.ylabel("Frequency")
      plt.xticks(rotation=90)
      plt.legend()
      plt.tight_layout()
      plt.show()
      
      # Plot final signal usage
      final_signal_usage = [histories[0]['signal_history'][-1],histories[1]['signal_history'][-1]]
      plt.figure(figsize=(8, 5))  # (width, height)
      for i, usage in enumerate(final_signal_usage):
          for state, counts in usage.items():
              total_counts = counts.sum()  # Normalize independently for each state
              proportions = counts / total_counts  # Normalize to proportions

              bar_labels = [f"{prop:.2f}" for prop in proportions]  # Format proportion labels
              bars = plt.bar(
                  [f"A{i}-{state}-Sig {s}" for s in range(n_signaling_actions)],
                  proportions,
                  label=f"A{i}, State {state}",
                  alpha=0.7
              )

              # Add proportion labels on top of each bar
              for bar, label in zip(bars, bar_labels):
                  plt.text(
                      bar.get_x() + bar.get_width() / 2,  # Center horizontally
                      bar.get_height(),                   # Position at the top of the bar
                      label,                              # The proportion label
                      ha='center',                        # Horizontal alignment
                      va='bottom'                         # Vertical alignment
                  )

      plt.title("Final Signal Usage Proportions by Observation")
      plt.ylabel("Proportion")
      plt.xticks(rotation=90)
      plt.legend()
      plt.tight_layout()
      plt.show()
      
      plt.figure(figsize=(8, 5))  # (width, height)
      # Dataset 1
      proportions1 = calculate_proportions(histories[0])
      for key, values in proportions1.items():
          smoothed_values = smooth(values)
          plt.plot(range(len(values)), smoothed_values, marker='o', markersize=1, label=f'Agent 0 - Key {key}')
          plt.text(len(values)-1, smoothed_values[-1], f'{smoothed_values[-1]:.2f}', fontsize=10, ha='right')
          
      # Dataset 2
      proportions2 = calculate_proportions(histories[1])
      for key, values in proportions2.items():
          smoothed_values = smooth(values)
          plt.plot(range(len(values)), smoothed_values, marker='x', markersize=1, label=f'Agent 1 - Key {key}')
          plt.text(len(values)-1, smoothed_values[-1], f'{smoothed_values[-1]:.2f}', fontsize=10, ha='right')
          
      plt.title('(Smoothed) Signal Urn Proportions History for Agent and Observation')
      plt.xlabel('Episode')
      plt.ylabel('Proportion')
      plt.grid(True)
      plt.legend()

    return signal_usage, rewards_history, signal_information_history, histories, env.nature_history
  


def temp_simulation_function(n_agents, n_features,
                        n_signaling_actions, n_final_actions,
                        n_episodes=6000, with_signals=True,
                        plot=True, env=None, verbose=False):

    for episode in range(n_episodes):
        if verbose:
            print(f'Episode {episode}')

        # Step 1: Sample nature and assign direct observations
        nature_vector = tuple(env.nature_sample())
        agents_observations = env.assign_observations(nature_vector)

        if verbose:
            print(f'Nature vector: {nature_vector}')
            print(f'Initial observations: {agents_observations}')

        # Step 2: Signal phase
        env.step_type = "signal"
        signals = env.get_actions(agents_observations)

        # Call play_step to get dummy reward and advance environment step
        signal_rewards, _ = env.play_step(signals)

        # Communication (optional)
        if with_signals:
            new_observations = env.communicate(agents_observations)
        else:
            new_observations = agents_observations[:]

        # Update agent's signaling Q-table
        env.update_agents(
            old_obs=agents_observations,
            actions=signals,
            rewards=signal_rewards,
            new_obs=new_observations,
            done=False
        )

        if verbose:
            print(f'Signals: {signals}')
            print(f'Post-communication observations: {new_observations}')

        # Step 3: Final action phase
        env.step_type = "act"
        final_actions = env.get_actions(new_observations)

        if verbose:
            print(f'Final actions: {final_actions}')

        # Step 4: Get rewards
        rewards, done = env.play_step(final_actions)

        # Step 5: Update agent knowledge
        # done=True here (end of episode)
        env.update_agents(
            old_obs=new_observations,
            actions=final_actions,
            rewards=rewards,
            new_obs=new_observations,  # or next obs in a multi-step setup
            done=True
        )

        if verbose:
            print(f'Rewards: {rewards}')

    signal_usage, rewards_history, signal_information_history, nature_history, histories = env.report_metrics()

    if plot:
      # Plot rewards over episodes
      plt.figure(figsize=(8, 5)) # (width, height)
      for i in range(n_agents):
        # first smoothing
        #smoothed_rewards = [sum(rewards_history[i][j:j+100]) / 100 for j in range(0, n_episodes, 100)]
        #plt.plot(range(0, n_episodes, 100), smoothed_rewards, label=f"Agent {i}")
        # second smoothing
        window_size = 100
        smoothed_rewards = np.convolve(rewards_history[i], np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, n_episodes), smoothed_rewards, label=f"Agent {i}")


      plt.title("Average Rewards (Smoothed)")
      plt.xlabel("Episode")
      plt.ylabel("Average Reward")
      plt.legend()

      # Plot NMI over episodes
      plt.figure(figsize=(8, 5)) # (width, height)
      for i in range(n_agents):
        smoothed_NMI = [sum(signal_information_history[i][j:j+10]) / 10 for j in range(0, n_episodes, 10)]
        plt.plot(range(0, n_episodes, 10), smoothed_NMI, label=f"Agent {i}")
      plt.title("Average Normalized Mutual Information (Smoothed)")
      plt.xlabel("Episode")
      plt.ylabel("Average NMI")
      plt.legend()

      # Plot total signal usage
      plt.figure(figsize=(8, 5)) # (width, height)
      for i, usage in enumerate(signal_usage):
          for state, counts in usage.items():
              bar_labels = [f"{count:.2f}" for count in counts]  # Format proportion labels
              bars = plt.bar(
                  [f"A{i}-{state}-Sig {s}" for s in range(n_signaling_actions)],
                  counts,
                  label=f"A{i}, State {state}",
                  alpha=0.7
              )

              # Add proportion labels on top of each bar
              for bar, label in zip(bars, bar_labels):
                  plt.text(
                      bar.get_x() + bar.get_width() / 2,  # Center horizontally
                      bar.get_height(),                   # Position at the top of the bar
                      label,                              # The proportion label
                      ha='center',                        # Horizontal alignment
                      va='bottom'                         # Vertical alignment
                  )
              # plt.bar(
              #     [f"A{i}-{state}-Sig {s}" for s in range(n_signaling_actions)],
              #     counts,
              #     label=f"A{i}, State {state}",
              #     alpha=0.7
              # )
      plt.title("Accumulated Signal Usage Count by Observation")
      plt.ylabel("Frequency")
      plt.xticks(rotation=90)
      plt.legend()
      plt.tight_layout()
      plt.show()
      
      # Plot final signal usage
      final_signal_usage = [histories[0]['signal_history'][-1],histories[1]['signal_history'][-1]]
      plt.figure(figsize=(8, 5))  # (width, height)
      for i, usage in enumerate(final_signal_usage):
          for state, counts in usage.items():
              total_counts = counts.sum()  # Normalize independently for each state
              proportions = counts / total_counts  # Normalize to proportions

              bar_labels = [f"{prop:.2f}" for prop in proportions]  # Format proportion labels
              bars = plt.bar(
                  [f"A{i}-{state}-Sig {s}" for s in range(n_signaling_actions)],
                  proportions,
                  label=f"A{i}, State {state}",
                  alpha=0.7
              )

              # Add proportion labels on top of each bar
              for bar, label in zip(bars, bar_labels):
                  plt.text(
                      bar.get_x() + bar.get_width() / 2,  # Center horizontally
                      bar.get_height(),                   # Position at the top of the bar
                      label,                              # The proportion label
                      ha='center',                        # Horizontal alignment
                      va='bottom'                         # Vertical alignment
                  )

      plt.title("Final Signal Usage Proportions by Observation")
      plt.ylabel("Proportion")
      plt.xticks(rotation=90)
      plt.legend()
      plt.tight_layout()
      plt.show()
      
      plt.figure(figsize=(8, 5))  # (width, height)
      # Dataset 1
      proportions1 = calculate_proportions(histories[0])
      for key, values in proportions1.items():
          smoothed_values = smooth(values)
          plt.plot(range(len(values)), smoothed_values, marker='o', markersize=1, label=f'Agent 0 - Key {key}')
          plt.text(len(values)-1, smoothed_values[-1], f'{smoothed_values[-1]:.2f}', fontsize=10, ha='right')
          
      # Dataset 2
      proportions2 = calculate_proportions(histories[1])
      for key, values in proportions2.items():
          smoothed_values = smooth(values)
          plt.plot(range(len(values)), smoothed_values, marker='x', markersize=1, label=f'Agent 1 - Key {key}')
          plt.text(len(values)-1, smoothed_values[-1], f'{smoothed_values[-1]:.2f}', fontsize=10, ha='right')
          
      plt.title('(Smoothed) Signal Urn Proportions History for Agent and Observation')
      plt.xlabel('Episode')
      plt.ylabel('Proportion')
      plt.grid(True)
      plt.legend()

    return signal_usage, rewards_history, signal_information_history, histories, nature_history
