from imports import *
from utils import *
from agents import UrnAgent
from environment import MultiAgentEnv

n_agents = 2
n_features = 2
n_signaling_actions = 2
n_final_actions = 4

random_game_dicts = {}
for i in range(n_agents):
  random_game_dicts[i] = create_randomcannonical_game(n_features,n_final_actions)

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
      urn_histories[i] = {'signal_urns_history':[],'action_urns_history':[]}

    nature_history = []

    for episode in range(n_episodes):
        if verbose:
          print(f'episode number is {episode}')
        # Reset the environment for a new episode
        nature_vector = tuple(env.reset())  # Convert nature vector to a tuple for Q-table indexing
        nature_history.append(nature_vector)
        if verbose:
          print(f'nature vector is {nature_vector}')
          
        # total_rewards = [0] * n_agents  # Track total rewards for each agent in the episode

        # Pre Step: Assign observations
        agents_observations = env.assign_observations()
        if verbose:
          print(f'agents direct observations are {agents_observations}')

        # Step 0: Agents choose signaling actions based on Q-learning policy
        signals = [agent.get_action(observation, is_signaling=True) for agent, observation in zip(agents, agents_observations)]
        # step to store signaling history, and move to the next step in the episode
        _ = env.step(signals)
        if verbose:
          print(f'agents signals are {signals}')
          print(f'environment step {env.current_step}')

        # Step 1: Agents choose final actions based on Q-learning policy
        # this step is different depending on whether agents they get each other's signals or not
        if with_signals:
          # Key here is that agent index 0 observes signal index 1 and viceversa
          new_observations = [agents_observations[0]+ (signals[1],), agents_observations[1]+(signals[0],)]
          #new_observations = [obs + (signal,) for obs, signal in zip(agents_observations,signals)]
          if verbose:
            print(f'agents new_observations are {new_observations}')
          final_actions = [agent.get_action(new_obs, is_signaling=False) for agent,new_obs in zip(agents,new_observations)]
        else: #
          final_actions = [agent.get_action(observation, is_signaling=False) for agent,observation in zip(agents,agents_observations)]

        rewards, done = env.step(final_actions)
        if verbose:
          print(f'agents final_actions are {final_actions}')
          print(f'environment step {env.current_step}')

        # Update Q-tables for signaling and final actions
        # update_urns(self, state, action, reward, is_signaling=True):
        for i, agent in enumerate(agents):
          # updating the signaling q_table
          if with_signals:
            # important that the state is the agents_observations and not the new observations
            # because this us updating the signal payoff, and the signal inputs are the initial observations
            agent.update(agents_observations[i], signals[i], rewards[i], is_signaling=True)
            # now we update the action q_table, the input being the new_observations
            agent.update(new_observations[i], final_actions[i], rewards[i], is_signaling=False)
          else: # if with_signals = False then there is no updating of signal q_table, but yes for action q_table
            agent.update(agents_observations[i], final_actions[i], rewards[i], is_signaling=False)

          if verbose:
            print(f'agent {i} signalling_urns are {agent.signalling_urns}')
            print(f'agent {i} action_urns are {agent.action_urns}')

        # Update urn histories
        # copy.deepcopy() is a function in Python's copy module that creates a deep copy of an object.
        # A deep copy means that the new object is a completely independent copy of the original,
        # including any nested objects it contains.
        for i, agent in enumerate(agents):
          urn_histories[i]['signal_urns_history'].append(copy.deepcopy(agent.signalling_urns))
          urn_histories[i]['action_urns_history'].append(copy.deepcopy(agent.action_urns))

        if verbose:
          print('Episode ended')
          print('\n')

    signal_usage, rewards_history, signal_information_history = env.report_metrics()

    if plot:
      # Plot results
      plt.figure(figsize=(8, 15)) # (width, height)

      # Plot rewards over episodes
      #plt.subplot(3, 1, 1)  # 3 rows, 1 column, index 1
      plt.figure(figsize=(8, 4)) # (width, height)
      for i in range(n_agents):
          smoothed_rewards = [sum(rewards_history[i][j:j+100]) / 100 for j in range(0, n_episodes, 100)]
          plt.plot(range(0, n_episodes, 100), smoothed_rewards, label=f"Agent {i}")
      plt.title("Average Rewards (Smoothed over 100 episodes)")
      plt.xlabel("Episode")
      plt.ylabel("Average Reward")
      plt.legend()

      # Plot NMI over episodes
      #plt.subplot(3, 1, 2)  # 3 rows, 1 column, index 2
      plt.figure(figsize=(8, 4)) # (width, height)
      for i in range(n_agents):
          smoothed_NMI = [sum(signal_information_history[i][j:j+10]) / 10 for j in range(0, n_episodes, 10)]
          plt.plot(range(0, n_episodes, 10), smoothed_NMI, label=f"Agent {i}")
      plt.title("Average Normalized Mutual Information (Smoothed over 10 episodes)")
      plt.xlabel("Episode")
      plt.ylabel("Average NMI")
      plt.legend()

      # Plot signal usage
      #plt.subplot(3, 1, 3)  # 3 rows, 1 column, index 3\
      plt.figure(figsize=(8, 5)) # (width, height)
      for i, usage in enumerate(signal_usage):
          for state, counts in usage.items():
              plt.bar(
                  [f"A{i}-{state}-Sig {s}" for s in range(n_signaling_actions)],
                  counts,
                  label=f"A{i}, State {state}",
                  alpha=0.7
              )
      plt.title("Signal Usage by Observation")
      plt.ylabel("Frequency")
      plt.xticks(rotation=90)
      plt.legend()
      plt.tight_layout()
      plt.show()

    return signal_usage, rewards_history, signal_information_history, urn_histories, nature_history