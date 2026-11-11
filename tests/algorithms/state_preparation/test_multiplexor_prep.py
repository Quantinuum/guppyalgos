"""Tests for multiplexor-based state preparation."""

from math import ceil, log2
from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import control, dagger
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, h
from selene_sim import Quest

from guppyalgos.algorithms.state_preparation.multiplexor_prep import (
    _multiplexed_rotation_commands,
    _state_preparation_angles,
    multiplexor_prep,
)
from guppyalgos.utils import qarray
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_statevector,
    get_total_state_on_only_specified_registers,
    switch_endianness,
)


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


def test_multiplexor_prep_controlled_relative_phase() -> None:
    """A complex prepared state retains its phase relative to an idle control branch."""
    state = np.array([1.0, 1j], dtype=np.complex128) / np.sqrt(2)
    prepare = multiplexor_prep(state)

    @guppy
    @no_type_check
    def main() -> None:
        controls = qarray(1)
        targets = qarray(1)
        h(controls[0])
        with control(controls[0]):
            prepare(targets)
        state_output("controls", controls)
        state_output("targets", targets)
        discard_array(controls)
        discard_array(targets)

    expected_target = np.array(
        [np.exp(-0.25j * np.pi), np.exp(0.25j * np.pi)], dtype=np.complex128
    ) / np.sqrt(2)
    expected = np.array(
        [
            1 / np.sqrt(2),
            expected_target[0] / np.sqrt(2),
            0.0,
            expected_target[1] / np.sqrt(2),
        ],
        dtype=np.complex128,
    )
    result = main.emulator(2).run()
    states = Quest.extract_states_dict(result.results[0].entries)
    actual, _ = get_total_state_on_only_specified_registers(
        states, ["controls", "targets"]
    )
    np.testing.assert_allclose(actual.state, expected, atol=1e-8)


def test_multiplexor_prep_daggered_roundtrip() -> None:
    """The daggered multiplexor unprepares its state back to zero."""
    prepare = multiplexor_prep(np.array([1.0, 1j]) / np.sqrt(2))

    @guppy
    @no_type_check
    def main() -> None:
        targets = qarray(1)
        prepare(targets)
        with dagger:
            prepare(targets)
        state_output("result_state", targets)
        discard_array(targets)

    actual = switch_endianness(get_statevector(main, 1))
    np.testing.assert_allclose(actual, np.array([1.0, 0.0]), atol=1e-8)
