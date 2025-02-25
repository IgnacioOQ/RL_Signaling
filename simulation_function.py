from imports import *
from utils import *
from agents import UrnAgent, QLearningAgent
from environment import MultiAgentEnv

n_agents = 2
n_features = 2
n_signaling_actions = 2
n_final_actions = 4

random_game_dicts = {}
for i in range(n_agents):
  random_game_dicts[i] = create_random_canonical_game(n_features,n_final_actions)

agents_observed_variables = {0:[0],1:[1]}

env = MultiAgentEnv(n_agents=n_agents, n_features=n_features,
                    n_signaling_actions=n_signaling_actions,
                    n_final_actions=n_final_actions,
                    full_information = False,
                    game_dicts=random_game_dicts,
                    observed_variables = agents_observed_variables)

def simulation_function(n_agents=n_agents, n_features=n_features,
                        n_signaling_actions=n_signaling_actions, n_final_actions=n_final_actions,
                        n_episodes=6000, with_signals = True,
                        plot=True,env=env,agent_type=UrnAgent,
                        initialize = False,
                        verbose=False):

    agents = [agent_type(n_signaling_actions, n_final_actions,
                       initialize=initialize) for _ in range(n_agents)]

    # History of the information status of the agent after each episode
    # namely their signalling and action urns as they get more complex
    urn_histories = {}
    for i, agent in enumerate(agents):
      urn_histories[i] = {'signal_history':[],'action_history':[]}

    nature_history = []

    for episode in range(n_episodes):
      if verbose:
        print(f'episode number is {episode}')
      # Reset the environment for a new episode
      nature_vector = tuple(env.reset())  # Convert nature vector to a tuple for Q-table indexing
      nature_history.append(nature_vector)
      if verbose:
        print(f'nature vector is {nature_vector}')

      # Pre Step: Assign observations
      agents_observations = env.assign_observations(nature_vector)
      if verbose:
        print(f'environment step {env.current_step}')
        print(f'agents direct observations are {agents_observations}')

      # Step 0: Agents choose signaling actions based on Q-learning policy
      signals = [agent.get_signal(observation) for agent, observation in zip(agents, agents_observations)]
      # step to store signaling history, and move to the next step in the episode
      _ = env.signals_step(signals,nature_vector)
      if verbose:
        print(f'agents signals are {signals}')

      # Step 1: Agents choose final actions based on Q-learning policy
      # this step is different depending on whether agents they get each other's signals or not
      if with_signals:
        # Key here is that agent index 0 observes signal index 1 and viceversa
        new_observations = [agents_observations[0]+ (signals[1],), agents_observations[1]+(signals[0],)]
        #new_observations = [obs + (signal,) for obs, signal in zip(agents_observations,signals)]
        if verbose:
          print(f'environment step {env.current_step}')
          print(f'agents new_observations are {new_observations}')
        final_actions = [agent.get_action(new_obs) for agent,new_obs in zip(agents,new_observations)]
      else: #
        final_actions = [agent.get_action(observation) for agent,observation in zip(agents,agents_observations)]

      rewards, done = env.actions_step(final_actions)
      if verbose:
        print(f'agents final_actions are {final_actions}')


      # Update Q-tables for signaling and final actions
      # update_urns(self, state, action, reward, is_signaling=True):
      for i, agent in enumerate(agents):
        # updating the signaling q_table
        if with_signals:
          # important that the state is the agents_observations and not the new observations
          # because this us updating the signal payoff, and the signal inputs are the initial observations
          agent.update_signals(agents_observations[i], signals[i], rewards[i])
          # now we update the action q_table, the input being the new_observations
          agent.update_actions(new_observations[i], final_actions[i], rewards[i])
        else: # if with_signals = False then there is no updating of signal q_table, but yes for action q_table
          agent.update_actions(agents_observations[i], final_actions[i], rewards[i])

        if verbose:
          if agent_type == UrnAgent:
            print(f'agent {i} signalling_urns are {agent.signalling_urns}')
            print(f'agent {i} action_urns are {agent.action_urns}')
          if agent_type == QLearningAgent:
              print(f'agent {i} signalling_counts are {agent.signalling_counts}')
              print(f'agent {i} action_counts are {agent.action_counts}')

      # Update urn histories
      # copy.deepcopy() is a function in Python's copy module that creates a deep copy of an object.
      # A deep copy means that the new object is a completely independent copy of the original,
      # including any nested objects it contains.
      if agent_type == UrnAgent:
        for i, agent in enumerate(agents):
          urn_histories[i]['signal_history'].append(copy.deepcopy(agent.signalling_urns))
          urn_histories[i]['action_history'].append(copy.deepcopy(agent.action_urns))
      if agent_type == QLearningAgent:
        for i, agent in enumerate(agents):
          urn_histories[i]['signal_history'].append(copy.deepcopy(agent.signalling_counts))
          urn_histories[i]['action_history'].append(copy.deepcopy(agent.action_counts))

      if verbose:
        print('Episode ended')
        print('\n')

    signal_usage, rewards_history, signal_information_history = env.report_metrics()

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
      final_signal_usage = [urn_histories[0]['signal_history'][-1],urn_histories[1]['signal_history'][-1]]
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
      proportions1 = calculate_proportions(urn_histories[0])
      for key, values in proportions1.items():
          smoothed_values = smooth(values)
          plt.plot(range(len(values)), smoothed_values, marker='o', markersize=1, label=f'Agent 0 - Key {key}')
          plt.text(len(values)-1, smoothed_values[-1], f'{smoothed_values[-1]:.2f}', fontsize=10, ha='right')
          
      # Dataset 2
      proportions2 = calculate_proportions(urn_histories[1])
      for key, values in proportions2.items():
          smoothed_values = smooth(values)
          plt.plot(range(len(values)), smoothed_values, marker='x', markersize=1, label=f'Agent 1 - Key {key}')
          plt.text(len(values)-1, smoothed_values[-1], f'{smoothed_values[-1]:.2f}', fontsize=10, ha='right')
          
      plt.title('(Smoothed) Signal Urn Proportions History for Agent and Observation')
      plt.xlabel('Episode')
      plt.ylabel('Proportion')
      plt.grid(True)
      plt.legend()

    return signal_usage, rewards_history, signal_information_history, urn_histories, nature_history