"""Tests for controlled Pauli exponentials."""

from __future__ import annotations

from typing import no_type_check

from guppylang import guppy
from guppylang.std.angles import angle, pi
from guppylang.std.builtins import array, control
from guppylang.std.debug import state_output
from guppylang.std.quantum import crz, discard_array, h, qubit
import numpy as np
import pytest
import zixy.qubit.pauli as zqp

from guppyalgos.primitives.pauli.pauli_exp import pauli_exp
from guppyalgos.primitives.subroutines.ladders import (
    Ladder,
    CXLadderLinear,
    CXLadderLog,
)
from guppyalgos.utils import qarray
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_statevector,
    get_unitary_projected,
)
from tests.primitives.pauli.pauli_exp.pauli_exp_helpers import pauli_exp_matrix

REPRESENTATIVE_CONTROLLED_2Q_STRINGS = [
    "X0 Y1",
    "Z0 Z1",
    "I0 X1",
    "Y0 I1",
]

REPRESENTATIVE_CONTROLLED_3Q_STRINGS = [
    "X0 Y1 Z2",
    "I0 X1 Y2",
]

REPRESENTATIVE_CONTROLLED_4Q_STRINGS = [
    "X0 Y1 Z2 Y3",
]


def check_controlled_pauli_exp(
    pauli_string: zqp.String,
    n_state_qubits: int,
    cx_ladder: type[Ladder],
) -> None:
    """Test the controlled Pauli exponential on all control blocks."""
    pauli_g = pauli_exp(pauli_string, n_state_qubits, cx_ladder)

    theta = 0.7

    @guppy
    @no_type_check
    def main(
        ctrl_qreg: array[qubit, 1],
        state_qreg: array[qubit, n_state_qubits],
    ) -> None:
        with control(ctrl_qreg[0]):
            pauli_g(state_qreg, angle(theta))
        state_output("ctrl", ctrl_qreg)

    active_block = pauli_exp_matrix(
        pauli_string, n_state_qubits, theta, little_endian=True
    )
    identity_block = np.eye(2**n_state_qubits, dtype=np.complex128)
    zero_block = np.zeros_like(identity_block)

    block_00 = get_unitary_projected(
        main, n_state_qubits, {"ctrl": [False]}, {"ctrl": [False]}
    )
    block_11 = get_unitary_projected(
        main, n_state_qubits, {"ctrl": [True]}, {"ctrl": [True]}
    )
    # TODO: Compute the full unitary once and slice the control blocks instead of
    # reconstructing four projected unitaries separately.
    # needs #159
    block_01 = get_unitary_projected(
        main, n_state_qubits, {"ctrl": [False]}, {"ctrl": [True]}
    )
    block_10 = get_unitary_projected(
        main, n_state_qubits, {"ctrl": [True]}, {"ctrl": [False]}
    )

    assert_allclose_ignorephase(block_00, identity_block)
    assert_allclose_ignorephase(block_11, active_block)
    np.testing.assert_allclose(block_01, zero_block, atol=1e-8)
    np.testing.assert_allclose(block_10, zero_block, atol=1e-8)


def test_controlled_pauli_exp_single_qubit_z0_matches_crz_phase_kickback() -> None:
    """Check the one-qubit ``Z0`` gadget matches the canonical Hadamard kickback."""
    phi = 0.33
    theta = -2 * phi
    pauli_string = zqp.String.from_str("Z0", 1)
    pauli_g = pauli_exp(pauli_string, 1, CXLadderLog)

    @guppy
    @no_type_check
    def pauli_main() -> None:
        ctrl_qreg = qarray(1)
        state_qreg = qarray(1)

        h(ctrl_qreg[0])
        with control(ctrl_qreg[0]):
            pauli_g(state_qreg, angle(theta))

        state_output("result_state", ctrl_qreg[0], state_qreg[0])
        discard_array(ctrl_qreg)
        discard_array(state_qreg)

    @guppy
    @no_type_check
    def crz_main() -> None:
        ctrl_qreg = qarray(1)
        state_qreg = qarray(1)

        h(ctrl_qreg[0])
        crz(ctrl_qreg[0], state_qreg[0], -2 * pi * phi)

        state_output("result_state", ctrl_qreg[0], state_qreg[0])
        discard_array(ctrl_qreg)
        discard_array(state_qreg)

    pauli_state = get_statevector(pauli_main, 2)
    crz_state = get_statevector(crz_main, 2)
    expected_relative_phase = np.exp(1j * np.pi * phi)
    expected_state = np.array(
        [1.0, expected_relative_phase, 0.0, 0.0], dtype=np.complex128
    ) / np.sqrt(2)

    assert_allclose_ignorephase(pauli_state, expected_state)
    assert_allclose_ignorephase(crz_state, expected_state)
    np.testing.assert_allclose(
        pauli_state[1] / pauli_state[0], expected_relative_phase, atol=1e-8
    )
    np.testing.assert_allclose(
        crz_state[1] / crz_state[0], expected_relative_phase, atol=1e-8
    )


@pytest.mark.parametrize(
    ("p_str", "n_state_qubits", "cx_ladder"),
    [
        (pauli_str, n_qubits, cx_method)
        for pauli_str in REPRESENTATIVE_CONTROLLED_2Q_STRINGS
        for n_qubits in [2, 3]
        for cx_method in [CXLadderLinear]
    ],
)
def test_controlled_pauli_exp_2q(
    p_str: str,
    n_state_qubits: int,
    cx_ladder: type[Ladder],
) -> None:
    """Test 2-qubit controlled Pauli exponentials over various Pauli strings."""
    pauli_string = zqp.String.from_str(p_str, n_state_qubits)
    check_controlled_pauli_exp(pauli_string, n_state_qubits, cx_ladder)


@pytest.mark.parametrize(
    ("p_str", "n_state_qubits", "cx_ladder"),
    [
        (pauli_str, n_qubits, cx_method)
        for pauli_str in REPRESENTATIVE_CONTROLLED_3Q_STRINGS
        for n_qubits in [3]
        for cx_method in [CXLadderLinear]
    ],
)
def test_controlled_pauli_exp_3q(
    p_str: str,
    n_state_qubits: int,
    cx_ladder: type[Ladder],
) -> None:
    """Test 3-qubit controlled Pauli exponentials over various Pauli strings."""
    pauli_string = zqp.String.from_str(p_str, n_state_qubits)
    check_controlled_pauli_exp(pauli_string, n_state_qubits, cx_ladder)


@pytest.mark.parametrize(
    ("p_str", "n_state_qubits", "cx_ladder"),
    [
        (pauli_str, n_qubits, cx_method)
        for pauli_str in REPRESENTATIVE_CONTROLLED_4Q_STRINGS
        for n_qubits in [4]
        for cx_method in [CXLadderLinear]
    ],
)
def test_controlled_pauli_exp_4q(
    p_str: str,
    n_state_qubits: int,
    cx_ladder: type[Ladder],
) -> None:
    """Test 4-qubit controlled Pauli exponentials over various Pauli strings."""
    pauli_string = zqp.String.from_str(p_str, n_state_qubits)
    check_controlled_pauli_exp(pauli_string, n_state_qubits, cx_ladder)
