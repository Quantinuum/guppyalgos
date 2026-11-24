"""Test repeat-until-success Rz rotation with arbitrary angles."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.debug import state_output
from guppylang.std.quantum import angle, discard, h, qubit, s, sdg
from selene_sim import QuantumReplay, Quest

from guppyalgos.primitives.rotations import (
    dummy_theta_resource_state,
    repeat_until_success_rz,
)


@pytest.mark.parametrize("ang", [0.1, 0.3, 0.5, 0.9])
def test_repeat_until_success_rz_arbitrary(ang: float) -> None:
    """Test repeat-until-success Rz rotation angle.

    Currently just defined for angles in radians [0, pi/1].
    This test checks that the implemented repeat-until-success Rz rotation
    produces the expected rotation angle within a tolerance when applied
    to a qubit initially prepared in the |+> state and measured after applying
    the Rz rotation and Hadamard gates.

    Uses replay simulation to quickly test all possible outcomes up to 20 failures.

    Args:
        ang (float): angle in radians/pi

    """
    rus_rz = repeat_until_success_rz(dummy_theta_resource_state)

    @guppy
    @no_type_check
    def circ_rus() -> None:
        q = qubit()
        theta = ang
        sdg(q)
        h(q)
        rus_rz(q, angle(theta))
        h(q)
        s(q)
        state_output("result_state", q)
        discard(q)

    n_shots = 20
    # attempts fail n times and then succeed
    desired_measurements = [[False] * n + [True] for n in range(0, n_shots)]

    rus_replay_sim = QuantumReplay(simulator=Quest(), measurements=desired_measurements)
    em_result = (
        circ_rus.emulator(2).with_simulator(rus_replay_sim).with_shots(n_shots).run()
    )
    # assert that the correct angle is found for every number of failures
    for shot_result in em_result.results:
        states = Quest.extract_states_dict(shot_result)
        sv = states["result_state"].state
        res = abs(sv[3]) ** 2
        theta = 2 * np.arcsin(np.sqrt(res)) / np.pi
        np.testing.assert_allclose(ang, theta)
