"""Tests for multiplexor-based state preparation."""

from math import ceil, log2
from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array

from guppyalgos.algorithms.state_preparation.multiplexor_prep import (
    _multiplexed_rotation_commands,
    _state_preparation_angles,
    multiplexor_prep,
)
from guppyalgos.utils import qarray
from guppyalgos.testing import assert_allclose_ignorephase, get_statevector


@pytest.mark.parametrize(
    ("state_size", "seed"), [(2, 1), (3, 2), (4, 3), (5, 4), (8, 5), (16, 6)]
)
def test_multiplexor_prep(state_size: int, seed: int) -> None:
    """Test multiplexor preparation on arbitrary normalized statevectors."""
    rng = np.random.default_rng(seed)
    state = rng.normal(size=state_size) + 1j * rng.normal(size=state_size)

    with pytest.raises(ValueError, match="normalized"):
        multiplexor_prep(state)

    state /= np.linalg.norm(state)
    n_qubits = ceil(log2(state_size))

    padded_state = np.pad(state, (0, 2**n_qubits - state_size))
    synthesis_state = padded_state.reshape((2,) * n_qubits).transpose().reshape(-1)
    _, y_angles, z_angles = _state_preparation_angles(synthesis_state)
    for angles in (y_angles, z_angles):
        commands = _multiplexed_rotation_commands(angles)
        expected_command_count = 1 if len(angles) == 1 else 2 * len(angles)
        assert len(commands) == expected_command_count

    state_prep = multiplexor_prep(state)

    @guppy
    @no_type_check
    def main() -> None:
        qs = qarray(n_qubits)
        state_prep(qs)
        state_output("result_state", qs)
        discard_array(qs)

    simulated_state = get_statevector(main, n_qubits)
    expected_state = np.pad(state, (0, 2**n_qubits - state_size))
    assert_allclose_ignorephase(simulated_state, expected_state, threshold=1e-7)
