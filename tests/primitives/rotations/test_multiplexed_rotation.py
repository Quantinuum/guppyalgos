"""Tests for uniformly controlled axis rotations."""

from typing import Any, no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import comptime
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard, discard_array, h, qubit, x
from selene_sim import Quest

from guppyalgos.primitives.rotations import (
    RotationAxisX,
    RotationAxisY,
    RotationAxisZ,
    multiplexed_rotation,
)
from guppyalgos.utils import qarray, int_to_bits, apply_bitstring
from tests.helpers.test_helpers import assert_allclose_ignorephase


def _expected_state(theta: float, axis: str) -> np.ndarray:
    phase = np.pi * theta / 2.0
    if axis == "x":
        return np.array([np.cos(phase), -1j * np.sin(phase)])
    if axis == "y":
        return np.array([np.cos(phase), np.sin(phase)])
    return np.array([np.exp(-1j * phase), 0.0])


@pytest.mark.parametrize(
    ("axis", "axis_type"),
    [("x", RotationAxisX), ("y", RotationAxisY), ("z", RotationAxisZ)],
)
def test_multiplexed_rotation_axes(axis: str, axis_type: Any) -> None:
    """A one-control rotation selects the expected X, Y, or Z rotation."""
    theta = 0.5

    @guppy
    @no_type_check
    def main() -> None:
        controls = qarray(1)
        x(controls[0])
        target = qubit()
        multiplexed_rotation(axis_type(), comptime([0.0, theta]), controls, target)
        state_output("target", target)
        discard_array(controls)
        discard(target)

    result = main.emulator(n_qubits=2).run()
    state = Quest.extract_states_dict(result.results[0].entries)[
        "target"
    ].get_single_state()
    assert_allclose_ignorephase(state, _expected_state(theta, axis))


def test_multiplexed_rotation_uses_little_endian_controls() -> None:
    """The first control selects angle index one in a two-control register."""
    theta = 0.5

    @guppy
    @no_type_check
    def main() -> None:
        controls = qarray(2)
        x(controls[0])
        target = qubit()
        multiplexed_rotation(
            RotationAxisY(), comptime([0.0, theta, 0.0, 0.0]), controls, target
        )
        state_output("target", target)
        discard_array(controls)
        discard(target)

    result = main.emulator(n_qubits=3).run()
    state = Quest.extract_states_dict(result.results[0].entries)[
        "target"
    ].get_single_state()
    assert_allclose_ignorephase(state, _expected_state(theta, "y"))


def test_multiplexed_rotation_control_superposition() -> None:
    """Preserve coherence when selecting rotations from a control superposition."""
    theta = 0.5

    @guppy
    @no_type_check
    def main() -> None:
        controls = qarray(1)
        h(controls[0])
        target = qubit()
        multiplexed_rotation(RotationAxisY(), comptime([0.0, theta]), controls, target)
        state_output("result_state", target, controls[0])
        discard_array(controls)
        discard(target)

    result = main.emulator(n_qubits=2).run()
    state = Quest.extract_states_dict(result.results[0].entries)[
        "result_state"
    ].get_single_state()
    phase = np.pi * theta / 2.0
    expected_state = np.array(
        [1.0, np.cos(phase), 0.0, np.sin(phase)], dtype=np.complex128
    ) / np.sqrt(2.0)
    assert_allclose_ignorephase(state, expected_state)


@pytest.mark.parametrize(
    ("control_index", "angles"),
    [
        (0, [0.25, 0.5]),
        (1, [0.125, 0.375, 0.5, 0.75]),
        (3, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]),
    ],
)
def test_multiplexed_rotation_applies_correct_angle(
    control_index: int, angles: list[float]
) -> None:
    """Check that the correct rotation angle is applied."""
    num_controls = int(np.floor(np.log2(len(angles))))
    bitstring = int_to_bits(control_index, num_controls)

    @guppy
    @no_type_check
    def main() -> None:
        controls = qarray(num_controls)
        target = qubit()
        apply_bitstring(controls, bitstring)
        multiplexed_rotation(RotationAxisY(), comptime(angles), controls, target)
        state_output("target", target)
        discard_array(controls)
        discard(target)

    result = main.emulator(n_qubits=num_controls + 1).run()
    state = Quest.extract_states_dict(result.results[0].entries)[
        "target"
    ].get_single_state()
    assert_allclose_ignorephase(state, _expected_state(angles[control_index], "y"))


def test_multiplexed_rotation_rejects_too_many_angles() -> None:
    """Reject an angle table with more entries than control basis states."""

    @guppy
    @no_type_check
    def main() -> None:
        controls = qarray(1)
        target = qubit()
        multiplexed_rotation(
            RotationAxisY(), comptime([0.0, 0.5, 1.0]), controls, target
        )
        discard_array(controls)
        discard(target)

    with pytest.raises(ValueError, match="n_angles == 2\\*\\*n_controls"):
        main.compile()
