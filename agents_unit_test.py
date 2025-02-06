from imports import *
from agents import UrnAgent, QLearningAgent

# Mocking create_initial_signals to avoid missing dependency errors
def create_initial_signals(n_observed_features, n_signals, n, m):
    return {f'state_{i}': np.ones(n_signals) for i in range(n_observed_features)}
class TestAgents(unittest.TestCase):

    def setUp(self):
        self.urn_agent = UrnAgent(n_signaling_actions=3, n_final_actions=2)
        self.q_agent = QLearningAgent(n_signaling_actions=3, n_final_actions=2)
        print(f"\nStarting test: {self._testMethodName}")

    def tearDown(self):
        print(f"Finished test: {self._testMethodName}\n")

    def test_urn_agent_initialization(self):
        try:
            self.assertEqual(self.urn_agent.n_signaling_actions, 3)
            self.assertEqual(self.urn_agent.n_final_actions, 2)
            self.assertEqual(self.urn_agent.signalling_urns, {})
            self.assertEqual(self.urn_agent.action_urns, {})
        except AssertionError as e:
            print(f"test_urn_agent_initialization failed: {e}")
            raise

    def test_q_learning_agent_initialization(self):
        try:
            self.assertEqual(self.q_agent.n_signaling_actions, 3)
            self.assertEqual(self.q_agent.n_final_actions, 2)
            self.assertEqual(self.q_agent.q_table_signaling, {})
            self.assertEqual(self.q_agent.q_table_action, {})
        except AssertionError as e:
            print(f"test_q_learning_agent_initialization failed: {e}")
            raise

    def test_urn_agent_get_action(self):
        try:
            action = self.urn_agent.get_action(state='state_1')
            self.assertIn(action, [0, 1, 2])
        except AssertionError as e:
            print(f"test_urn_agent_get_action failed: {e}")
            raise

    def test_q_learning_agent_get_action(self):
        try:
            action = self.q_agent.get_action(state='state_1')
            self.assertIn(action, [0, 1, 2])
        except AssertionError as e:
            print(f"test_q_learning_agent_get_action failed: {e}")
            raise

    def test_urn_agent_update(self):
        try:
            # Test signaling update
            self.urn_agent.update(state='state_1', action=1, reward=5, is_signaling=True)
            self.assertEqual(self.urn_agent.signalling_urns['state_1'][1], 6)

            # Test final action update
            self.urn_agent.update(state='state_2', action=0, reward=3, is_signaling=False)
            self.assertEqual(self.urn_agent.action_urns['state_2'][0], 4)
        except AssertionError as e:
            print(f"test_urn_agent_update failed: {e}")
            raise

    def test_q_learning_agent_update(self):
        try:
            self.q_agent.update(state='state_1', action=1, reward=5)
            self.assertGreater(self.q_agent.q_table_signaling['state_1'][1], 0)
        except AssertionError as e:
            print(f"test_q_learning_agent_update failed: {e}")
            raise

    def test_urn_agent_reset(self):
        try:
            self.urn_agent.update(state='state_1', action=1, reward=5)
            self.urn_agent.reset_urns()
            self.assertEqual(self.urn_agent.signalling_urns, {})
            self.assertEqual(self.urn_agent.action_urns, {})
        except AssertionError as e:
            print(f"test_urn_agent_reset failed: {e}")
            raise

if __name__ == '__main__':
    unittest.main()