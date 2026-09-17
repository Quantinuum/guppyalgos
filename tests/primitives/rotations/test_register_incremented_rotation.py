"""Tests for register-controlled ancilla rotations."""

from typing import Any, no_type_check

import numpy as np
import pytest
from guppylang import comptime, guppy
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, x, qubit, discard
from selene_sim import Quest

from guppyalgos.primitives.rotations import (
    RotationAxisX,
    RotationAxisY,
    RotationAxisZ,
    RotationRegisterIncremented,
)
from guppyalgos.utils import qarray
from guppyalgos.testing import assert_allclose_ignorephase


def _axis_expected_state(theta: float, axis: str) -> np.ndarray:
    """Return the expected ancilla state after a total axis rotation `theta`.

    The implementation uses tket units, so a rotation value `theta` corresponds
    to physical angle `phi = pi * theta`. Acting on `|0>`, the target one-qubit
    states are:
      - X axis: `cos(phi/2)|0> - i sin(phi/2)|1>`
      - Y axis: `cos(phi/2)|0> + sin(phi/2)|1>`
      - Z axis: global phase on `|0>` only, `exp(-i phi/2)|0>`
    with `phi/2 = pi * theta / 2`.
    """
    phase = np.pi * theta / 2.0
    if axis == "x":
        return np.array([np.cos(phase), -1j * np.sin(phase)], dtype=np.complex128)
    if axis == "y":
        return np.array([np.cos(phase), np.sin(phase)], dtype=np.complex128)
    return np.array([np.exp(-1j * phase), 0.0], dtype=np.complex128)


@pytest.mark.parametrize(
    ("bits", "axis", "axis_type"),
    [
        ([True, False, True], "y", RotationAxisY),
        ([False, True, True], "x", RotationAxisX),
        ([True, True, False, False], "z", RotationAxisZ),
        ([True, True, True, True], "y", RotationAxisY),
        ([False, False, False], "x", RotationAxisX),
        ([False, True, False, False], "z", RotationAxisZ),
    ],
)
def test_register_incremented_rotation(
    bits: list[bool],
    axis: str,
    axis_type: Any,
) -> None:
    """Check full-range fixed-point angle kickback on the ancilla."""
    n_phase_qubits = len(bits)

    @guppy
    @no_type_check
    def main() -> None:
        idx = comptime(bits)
        phase_qreg = qarray(comptime(n_phase_qubits))
        rotation_target = qubit()

        for bit in range(comptime(n_phase_qubits)):
            if idx[bit]:
                x(phase_qreg[bit])

        rotation = RotationRegisterIncremented[comptime(n_phase_qubits), axis_type](
            axis_type()
        )
        rotation.compose(phase_qreg, rotation_target)

        state_output("ancilla", rotation_target)
        discard_array(phase_qreg)
        discard(rotation_target)

    res = main.emulator(n_qubits=n_phase_qubits + 1).run()
    states = Quest.extract_states_dict(res.results[0].entries)
    ancilla_state = states["ancilla"].get_single_state()

    encoded_integer = sum((2**i) for i, bit in enumerate(bits) if bit)
    expected_theta = float(2.0 * encoded_integer / (2**n_phase_qubits))
    expected_state = _axis_expected_state(expected_theta, axis)
    assert_allclose_ignorephase(ancilla_state, expected_state)
