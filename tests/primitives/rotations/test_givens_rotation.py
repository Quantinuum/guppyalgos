"""Tests for register-controlled Givens rotations."""

from typing import Literal, no_type_check

import numpy as np
import pytest
from guppylang import comptime, guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array
from guppylang.std.debug import state_result
from guppylang.std.quantum import discard_array, qubit, x
from selene_sim import Quest

from guppyalgos.primitives.rotations import (
    GivensRotationPhaseGradient,
    GivensRotationRegisterIncremented,
    givens_rotation,
)
from guppyalgos.primitives.state_preparation.phase_gradient import phase_gradient
from guppyalgos.utils import qarray
from guppyalgos.testing import Endianness, assert_allclose_ignorephase, get_unitary


def _expected_givens_state(theta: float, target_bits: tuple[bool, bool]) -> np.ndarray:
    """Return the expected two-qubit state after the Givens rotation."""
    phase = np.pi * theta
    cosine = np.cos(phase)
    sine = np.sin(phase)

    if target_bits == (False, False):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    if target_bits == (False, True):
        return np.array([0.0, cosine, -sine, 0.0], dtype=np.complex128)
    if target_bits == (True, False):
        return np.array([0.0, sine, cosine, 0.0], dtype=np.complex128)
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex128)


@pytest.mark.parametrize(
    "synthesis",
    ["register_incremented", "phase_gradient"],
)
@pytest.mark.parametrize(
    ("bits", "target_bits"),
    [
        ([True, False, True], (False, True)),
        ([False, True, True], (True, False)),
        ([True, True, False, False], (False, False)),
        ([True, True, True, True], (True, True)),
    ],
)
def test_givens_rotation(
    synthesis: Literal["register_incremented", "phase_gradient"],
    bits: list[bool],
    target_bits: tuple[bool, bool],
) -> None:
    """Check both Givens synthesis methods on computational-basis sectors."""
    n_phase_qubits = len(bits)
    use_phase_gradient = synthesis == "phase_gradient"
    prepare_phase_gradient = phase_gradient(n_phase_qubits)

    @guppy
    @no_type_check
    def main() -> None:
        idx = comptime(bits)
        data_qreg = qarray(comptime(n_phase_qubits))
        rot_regs = (qubit(), qubit())

        for bit in range(comptime(n_phase_qubits)):
            if idx[bit]:
                x(data_qreg[bit])

        if comptime(target_bits[0]):
            x(rot_regs[0])
        if comptime(target_bits[1]):
            x(rot_regs[1])

        if comptime(use_phase_gradient):
            phase_gradient_state = qarray(comptime(n_phase_qubits))
            prepare_phase_gradient(phase_gradient_state)
            phase_gradient_rotation = GivensRotationPhaseGradient[
                comptime(n_phase_qubits)
            ](phase_gradient_state)
            phase_gradient_rotation.compose(data_qreg, rot_regs)
            discard_array(phase_gradient_rotation.phase_gradient)
        else:
            register_incremented_rotation = GivensRotationRegisterIncremented[
                comptime(n_phase_qubits)
            ]()
            register_incremented_rotation.compose(data_qreg, rot_regs)

        state_result("target", rot_regs[0], rot_regs[1])
        discard_array(data_qreg)
        rot_regs[0].discard()
        rot_regs[1].discard()

    res = main.emulator(n_qubits=(3 * n_phase_qubits) + 2).run()
    states = Quest.extract_states_dict(res.results[0].entries)
    target_state = states["target"].get_single_state()

    encoded_integer = sum((2**i) for i, bit in enumerate(bits) if bit)
    expected_theta = float(2.0 * encoded_integer / (2**n_phase_qubits))
    expected_state = _expected_givens_state(expected_theta, target_bits)
    assert_allclose_ignorephase(target_state, expected_state)


def _expected_givens_unitary(theta: float) -> np.ndarray:
    """Return the occupation-space matrix for the elementary Givens gadget."""
    phase = np.pi * theta
    cosine = np.cos(phase)
    sine = np.sin(phase)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cosine, sine, 0.0],
            [0.0, -sine, cosine, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )


@pytest.mark.parametrize("theta", [0.23, -0.31])
def test_elementary_givens_rotation_matches_mode_matrix(theta: float) -> None:
    """Check the elementary Givens gadget on the complete occupation space."""

    @guppy
    @no_type_check
    def main(qreg: array[qubit, 2]) -> None:
        givens_rotation(qreg[0], qreg[1], angle(comptime(theta)))

    compiled = get_unitary(main, 2, endianness=Endianness.LITTLE)
    np.testing.assert_allclose(compiled, _expected_givens_unitary(theta), atol=1e-8)
