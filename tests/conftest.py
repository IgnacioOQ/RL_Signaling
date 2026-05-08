"""Test-suite fixtures and import-path setup.

Adds the repository root to ``sys.path`` so ``pytest tests/`` works without
running ``pip install -e .`` first. With the install, this is a no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def two_agent_graph() -> nx.DiGraph:
    """Fully-connected 2-node directed graph (the canonical test scaffold)."""
    g = nx.DiGraph()
    g.add_nodes_from([0, 1])
    g.add_edges_from([(0, 1), (1, 0)])
    return g


@pytest.fixture
def small_game_dicts() -> dict:
    """Deterministic 2-feature, 4-action canonical games per agent.

    Built without ``random`` so tests do not depend on the global RNG state.
    """
    # State -> {action -> reward}; one canonical optimum per state.
    canonical = {
        (0, 0): {0: 1, 1: 0, 2: 0, 3: 0},
        (0, 1): {0: 0, 1: 1, 2: 0, 3: 0},
        (1, 0): {0: 0, 1: 0, 2: 1, 3: 0},
        (1, 1): {0: 0, 1: 0, 2: 0, 3: 1},
    }
    return {0: dict(canonical), 1: dict(canonical)}
