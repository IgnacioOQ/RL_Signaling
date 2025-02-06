from imports import *
from utils import *
from agents import UrnAgent, QLearningAgent
from environment import MultiAgentEnv

class TestAgents(unittest.TestCase):

    def setUp(self):
        self.urn_agent = UrnAgent(n_signaling_actions=3, n_final_actions=3)
        self.q_agent = QLearningAgent(n_signaling_actions=3, n_final_actions=3)
        self.state = 'test_state'

    def test_urn_agent_initialization(self):
        self.assertEqual(self.urn_agent.n_signaling_actions, 3)
        self.assertEqual(self.urn_agent.n_final_actions, 3)
        self.assertEqual(self.urn_agent.signalling_urns, {})
        self.assertEqual(self.urn_agent.action_urns, {})

    def test_urn_agent_get_signal(self):
        signal = self.urn_agent.get_signal(self.state)
        self.assertIn(signal, range(3))
        self.assertIn(self.state, self.urn_agent.signalling_urns)

    def test_urn_agent_get_action(self):
        action = self.urn_agent.get_action(self.state)
        self.assertIn(action, range(3))
        self.assertIn(self.state, self.urn_agent.action_urns)

    def test_urn_agent_update_signals(self):
        self.urn_agent.get_signal(self.state)
        self.urn_agent.update_signals(self.state, 0, 1.0)
        self.assertEqual(self.urn_agent.signalling_urns[self.state][0], 2.0)

    def test_urn_agent_update_actions(self):
        self.urn_agent.get_action(self.state)
        self.urn_agent.update_actions(self.state, 1, 1.5)
        self.assertEqual(self.urn_agent.action_urns[self.state][1], 2.5)

    def test_q_agent_initialization(self):
        self.assertEqual(self.q_agent.n_signaling_actions, 3)
        self.assertEqual(self.q_agent.n_final_actions, 3)
        self.assertEqual(self.q_agent.q_table_signaling, {})
        self.assertEqual(self.q_agent.q_table_action, {})

    def test_q_agent_get_signal(self):
        signal = self.q_agent.get_signal(self.state)
        self.assertIn(signal, range(3))
        self.assertIn(self.state, self.q_agent.q_table_signaling)

    def test_q_agent_get_action(self):
        action = self.q_agent.get_action(self.state)
        self.assertIn(action, range(3))
        self.assertIn(self.state, self.q_agent.q_table_action)

    def test_q_agent_update_signals(self):
        self.q_agent.get_signal(self.state)
        self.q_agent.update_signals(self.state, 0, 1.0)
        self.assertAlmostEqual(self.q_agent.q_table_signaling[self.state][0], 0.05)

    def test_q_agent_update_actions(self):
        self.q_agent.get_action(self.state)
        self.q_agent.update_actions(self.state, 1, 2.0)
        self.assertAlmostEqual(self.q_agent.q_table_action[self.state][1], 0.1)

    def test_reset_methods(self):
        self.urn_agent.get_signal(self.state)
        self.urn_agent.get_action(self.state)
        self.urn_agent.reset_urns()
        self.assertEqual(self.urn_agent.signalling_urns, {})
        self.assertEqual(self.urn_agent.action_urns, {})

        self.q_agent.get_signal(self.state)
        self.q_agent.get_action(self.state)
        self.q_agent.reset()
        self.assertEqual(self.q_agent.q_table_signaling, {})
        self.assertEqual(self.q_agent.q_table_action, {})


class TestMultiAgentEnv(unittest.TestCase):

    def setUp(self):
        self.env = MultiAgentEnv(n_agents=2, n_features=2, n_signaling_actions=2, n_final_actions=4,
                                 full_information=True,
                                 game_dicts=[{(0, 1): {0: 1, 1: 2}, (1, 0): {2: 3, 3: 4}}, 
                                             {(0, 1): {0: 1, 1: 2}, (1, 0): {2: 3, 3: 4}}],
                                 observed_variables={0: [0], 1: [1]})

    def test_reset(self):
        nature_vector = self.env.reset()
        self.assertEqual(len(nature_vector), self.env.n_features)
        self.assertTrue(all(bit in [0, 1] for bit in nature_vector))

    def test_signals_step(self):
        self.env.reset()
        signals = [0, 1]
        nature_vector = [1, 0]
        status = self.env.signals_step(signals, nature_vector)
        self.assertFalse(status)

    def test_actions_step(self):
        self.env.reset()
        self.env.signals_step([0, 1], [1, 0])
        rewards, done = self.env.actions_step([2, 3])
        self.assertTrue(done)
        self.assertEqual(rewards, [3, 4])

    def test_invalid_signal(self):
        self.env.reset()
        with self.assertRaises(ValueError):
            self.env.signals_step([3, 1], [1, 0])  # Signal 3 is out of range

    def test_invalid_action(self):
        self.env.reset()
        self.env.signals_step([0, 1], [1, 0])
        with self.assertRaises(KeyError):
            self.env.actions_step([5, 6])  # Invalid actions

if __name__ == '__main__':
    unittest.main()
