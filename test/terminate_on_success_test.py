from __future__ import annotations

import numpy as np
import pytest

from mpe2 import simple_spread_v3


def _run_covering_landmarks_on_cycle(max_cycles: int, success_cycle: int):
    """Run simple_spread with no-op actions, covering every landmark before `success_cycle`."""
    env = simple_spread_v3.parallel_env(
        N=3, max_cycles=max_cycles, terminate_on_success=True
    )
    env.reset(seed=0)
    world = env.unwrapped.world
    cycles = 0
    while env.agents:
        if cycles == success_cycle - 1:
            for agent, landmark in zip(world.agents, world.landmarks):
                agent.state.p_pos = landmark.state.p_pos.copy()
                agent.state.p_vel = np.zeros(world.dim_p)
        _, _, terminations, truncations, _ = env.step({a: 0 for a in env.agents})
        cycles += 1
    env.close()
    return cycles, terminations, truncations


@pytest.mark.parametrize("max_cycles", [1, 2, 25])
def test_success_on_last_cycle_terminates(max_cycles: int) -> None:
    cycles, terminations, truncations = _run_covering_landmarks_on_cycle(
        max_cycles, success_cycle=max_cycles
    )

    assert cycles == max_cycles
    assert all(terminations.values())
    assert not any(truncations.values())


def test_success_before_last_cycle_terminates_early() -> None:
    cycles, terminations, truncations = _run_covering_landmarks_on_cycle(
        max_cycles=25, success_cycle=10
    )

    assert cycles == 10
    assert all(terminations.values())
    assert not any(truncations.values())


def test_no_success_is_truncated_at_max_cycles() -> None:
    cycles, terminations, truncations = _run_covering_landmarks_on_cycle(
        max_cycles=5, success_cycle=-1
    )

    assert cycles == 5
    assert not any(terminations.values())
    assert all(truncations.values())
