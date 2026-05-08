"""RL_Signaling — emergent signaling between RL agents under partial observation."""

__version__ = "0.1.0"

from rl_signaling.agents import (
    BaseAgent,
    QLearningAgent,
    TDLearningAgent,
    UrnAgent,
)
from rl_signaling.env import MultiAgentEnv, NetMultiAgentEnv, TempNetMultiAgentEnv
from rl_signaling.simulation import (
    run_simulation,
    simulation_function,
    temp_simulation_function,
)

__all__ = [
    # Canonical public API
    "BaseAgent",
    "MultiAgentEnv",
    "QLearningAgent",
    "TDLearningAgent",
    "UrnAgent",
    "run_simulation",
    # Deprecated — kept for backward compatibility with existing notebooks
    "NetMultiAgentEnv",
    "TempNetMultiAgentEnv",
    "simulation_function",
    "temp_simulation_function",
]
