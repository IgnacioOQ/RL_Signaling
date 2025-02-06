from imports import *
from utils import *
from agents import UrnAgent, QLearningAgent
from environment import MultiAgentEnv

class TestUrnAgent(unittest.TestCase):
    def setUp(self):
        self.agent = UrnAgent(n_signaling_actions=3, n_final_actions=2)

    def test_initialization(self):
        self.assertEqual(self.agent.n_signaling_actions, 3)
        self.assertEqual(self.agent.n_final_actions, 2)
        self.assertEqual(self.agent.signalling_urns, {})
        self.assertEqual(self.agent.action_urns, {})

    def test_get_signal(self):
        state = 'state_1'
        signal = self.agent.get_signal(state)
        self.assertIn(signal, range(3))
        self.assertIn(state, self.agent.signalling_urns)

    def test_get_action(self):
        state = 'state_2'
        action = self.agent.get_action(state)
        self.assertIn(action, range(2))
        self.assertIn(state, self.agent.action_urns)

    def test_update_signals(self):
        state = 'state_3'
        self.agent.update_signals(state, 1, 5)
        self.assertEqual(self.agent.signalling_urns[state][1], 6)

    def test_update_actions(self):
        state = 'state_4'
        self.agent.update_actions(state, 0, 3)
        self.assertEqual(self.agent.action_urns[state][0], 4)

    def test_reset_urns(self):
        self.agent.update_signals('state_5', 0, 2)
        self.agent.reset_urns()
        self.assertEqual(self.agent.signalling_urns, {})
        self.assertEqual(self.agent.action_urns, {})


class TestQLearningAgent(unittest.TestCase):
    def setUp(self):
        self.agent = QLearningAgent(n_signaling_actions=3, n_final_actions=2)

    def test_initialization(self):
        self.assertEqual(self.agent.n_signaling_actions, 3)
        self.assertEqual(self.agent.n_final_actions, 2)
        self.assertEqual(self.agent.q_table_signaling, {})
        self.assertEqual(self.agent.q_table_action, {})

    def test_get_signal(self):
        state = 'state_1'
        signal = self.agent.get_signal(state)
        self.assertIn(signal, range(3))
        self.assertIn(state, self.agent.q_table_signaling)

    def test_get_action(self):
        state = 'state_2'
        action = self.agent.get_action(state)
        self.assertIn(action, range(2))
        self.assertIn(state, self.agent.q_table_action)

    def test_update_signals(self):
        state = 'state_3'
        self.agent.update_signals(state, 1, 5)
        self.assertAlmostEqual(self.agent.q_table_signaling[state][1], 0.25)

    def test_update_actions(self):
        state = 'state_4'
        self.agent.update_actions(state, 0, 3)
        self.assertAlmostEqual(self.agent.q_table_action[state][0], 0.15)

    def test_reset(self):
        self.agent.update_signals('state_5', 0, 2)
        self.agent.reset()
        self.assertEqual(self.agent.q_table_signaling, {})
        self.assertEqual(self.agent.q_table_action, {})


class TestMultiAgentEnv(unittest.TestCase):
    def setUp(self):
        """Set up the environment for testing."""
        self.env = MultiAgentEnv(n_agents=2, n_features=2, n_signaling_actions=2, n_final_actions=4,
                                 full_information=True, 
                                 game_dicts=[{(0, 0): {0: 1, 1: 2}, (1, 1): {2: 3, 3: 4}},
                                             {(0, 0): {0: 1, 1: 2}, (1, 1): {2: 3, 3: 4}}])

    def test_reset(self):
        """Test the reset functionality."""
        nature_vector = self.env.reset()
        self.assertEqual(len(nature_vector), self.env.n_features)
        self.assertTrue(all(bit in [0, 1] for bit in nature_vector))

    def test_step_signaling(self):
        """Test the signaling step."""
        self.env.reset()
        actions = [0, 1]
        result = self.env.step(actions)
        self.assertFalse(result)  # Expecting False since only signaling is done
        self.assertEqual(self.env.signals, actions)

    def test_step_final_actions(self):
        """Test the final action step and reward calculation."""
        self.env.reset()
        self.env.step([0, 1])  # Signaling step

        # Ensure nature_vector matches a valid key in the game_dicts
        self.env.nature_vector = np.array([0, 0])  # Override for deterministic test
        actions = [0, 1]
        rewards, done = self.env.step(actions)

        self.assertTrue(done)
        self.assertEqual(rewards, [1, 2])
        self.assertEqual(self.env.rewards_history, [[1], [2]])

    def test_invalid_signal(self):
        """Test handling of invalid signaling action."""
        self.env.reset()
        with self.assertRaises(ValueError):
            self.env.step([5, 1])  # Invalid signal for agent 0

    def test_invalid_final_action(self):
        """Test handling of invalid final action."""
        self.env.reset()
        self.env.step([0, 1])  # Valid signaling
        self.env.nature_vector = np.array([0, 0])  # Ensure consistent state
        with self.assertRaises(KeyError):
            self.env.step([3, 4])  # Invalid final action

    def test_report_metrics(self):
        """Test the reporting of metrics."""
        self.env.reset()
        self.env.step([0, 1])
        self.env.nature_vector = np.array([0, 0])
        self.env.step([0, 1])

        signal_usage, rewards_history, signal_information_history = self.env.report_metrics()
        self.assertIsInstance(signal_usage, list)
        self.assertIsInstance(rewards_history, list)
        self.assertIsInstance(signal_information_history, list)


if __name__ == '__main__':
    unittest.main()
