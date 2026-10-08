"""Pauli Exponential tests."""

from __future__ import annotations
from guppylang import guppy

from guppylang.std.builtins import array, control, dagger
from guppylang.std.angles import angle
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, h, qubit, rz
import numpy as np
import pytest

from guppyalgos.primitives.subroutines.ladders import CXLadderLinear, Ladder


import zixy.qubit.pauli as zqp

from guppylang.defs import GuppyFunctionDefinition

from guppyalgos.primitives.pauli.pauli_exp import pauli_exp

from guppyalgos.testing import get_statevector, get_unitary
from tests.primitives.pauli.pauli_exp.pauli_exp_helpers import pauli_exp_matrix
from guppyalgos.testing import assert_allclose_ignorephase
from guppyalgos.utils import qarray
from typing import no_type_check

REPRESENTATIVE_2Q_STRINGS = [
    "X0 Y1",
    "Z0 Z1",
    "I0 X1",
    "Y0 I1",
]

REPRESENTATIVE_3Q_STRINGS = [
    "X0 Y1 Z2",
    "Y0 Y1 Y2",
    "I0 X1 Y2",
    "Y0 I1 X2",
]

REPRESENTATIVE_4Q_STRINGS = [
    "X0 Y1 Z2 Y3",
]


def pauli_exp_test_fn(
    pauli_string: zqp.String,
    n_state_qubits: int,
    cx_ladder: type[Ladder],
    rz_method: GuppyFunctionDefinition[[qubit, angle], None],
) -> None:
    """Test the pauli exponential implementation against the matrix exponential.

    This test function constructs the pauli exponential guppy function parameterized
    over the input pauli string, number of qubits, CX ladder method and RZ method. It
    then compares the unitary matrix obtained from simulating the guppy function against
    the matrix exponential of the pauli string.

    The edge case it tests is that of identity pauli operators in the string, which
    should be handled correctly by the basis change and CX ladder procedures. Also when
    the pauli string acts on less than the full qubit register.

    Args:
        pauli_string (zqp.String): zixy String representing the Pauli to be
            exponentiated
        n_state_qubits: number of qubits in the register to apply the Pauli exponential
        cx_ladder (Ladder): CX ladder implementing Ladder protocol.
        rz_method (GuppyFunctionDefinition): RZ decomposition method to use for the R

    """
    pauli_g = pauli_exp(pauli_string, n_state_qubits, cx_ladder, rz_method)

    theta = 0.7

    @guppy
    @no_type_check
    def main(state_qreg: array[qubit, n_state_qubits]) -> None:
        pauli_g(state_qreg, angle(theta))

    guppy_u = get_unitary(main, n_state_qubits)
    u_mat = pauli_exp_matrix(pauli_string, n_state_qubits, theta, little_endian=True)

    assert_allclose_ignorephase(u_mat, guppy_u)

    @guppy
    @no_type_check
    def main_dagger(state_qreg: array[qubit, n_state_qubits]) -> None:
        with dagger:
            pauli_g(state_qreg, angle(theta))

    dagger_u = get_unitary(main_dagger, n_state_qubits)
    assert_allclose_ignorephase(u_mat.conj().T, dagger_u)


@pytest.mark.parametrize(
    ("p_str", "n_state_qubits", "cx_ladder", "rz_method"),
    [
        (pauli_str, n_qubits, cx_method, rz_method)
        for pauli_str in REPRESENTATIVE_2Q_STRINGS
        for n_qubits in [2, 3]
        for cx_method in [CXLadderLinear]
        for rz_method in [rz]
    ],
)
def test_pauli_exp_2q(
    p_str: str,
    n_state_qubits: int,
    cx_ladder: type[Ladder],
    rz_method: GuppyFunctionDefinition[[qubit, angle], None],
) -> None:
    """Test 2-qubit pauli exponentials over various pauli strings and methods."""
    pauli_string = zqp.String.from_str(p_str, n_state_qubits)
    pauli_exp_test_fn(pauli_string, n_state_qubits, cx_ladder, rz_method)


@pytest.mark.parametrize(
    ("p_str", "n_state_qubits", "cx_ladder", "rz_method"),
    [
        (pauli_str, n_qubits, cx_method, rz_method)
        for pauli_str in REPRESENTATIVE_3Q_STRINGS
        for n_qubits in [3]
        for cx_method in [CXLadderLinear]
        for rz_method in [rz]
    ],
)
def test_pauli_exp_3q(
    p_str: str,
    n_state_qubits: int,
    cx_ladder: type[Ladder],
    rz_method: GuppyFunctionDefinition[[qubit, angle], None],
) -> None:
    """Test 3-qubit pauli exponentials over various pauli strings and methods."""
    pauli_string = zqp.String.from_str(p_str, n_state_qubits)
    pauli_exp_test_fn(pauli_string, n_state_qubits, cx_ladder, rz_method)


@pytest.mark.parametrize(
    ("p_str", "n_state_qubits", "cx_ladder", "rz_method"),
    [
        (pauli_str, n_qubits, cx_method, rz_method)
        for pauli_str in REPRESENTATIVE_4Q_STRINGS
        for n_qubits in [4]
        for cx_method in [CXLadderLinear]
        for rz_method in [rz]
    ],
)
def test_pauli_exp_4q(
    p_str: str,
    n_state_qubits: int,
    cx_ladder: type[Ladder],
    rz_method: GuppyFunctionDefinition[[qubit, angle], None],
) -> None:
    """Test 4-qubit pauli exponentials over various pauli strings and methods."""
    pauli_string = zqp.String.from_str(p_str, n_state_qubits)
    pauli_exp_test_fn(pauli_string, n_state_qubits, cx_ladder, rz_method)


def test_pauli_exp_identity_is_noop() -> None:
    """An uncontrolled identity exponential is a no-op up to global phase."""
    pauli_string = zqp.String.from_str("I0 I1 I2", 3)
    pauli_g = pauli_exp(pauli_string, 3, CXLadderLinear, rz)

    @guppy
    @no_type_check
    def main(state_qreg: array[qubit, 3]) -> None:
        pauli_g(state_qreg, angle(0.7))

    np.testing.assert_allclose(get_unitary(main, 3), np.eye(8), atol=1e-8)


def test_pauli_exp_controlled_and_ctrl_daggered() -> None:
    """Controlled modifier variants preserve the expected relative phase."""
    theta = 0.7
    pauli_string = zqp.String.from_str("Z0", 1)
    pauli_g = pauli_exp(pauli_string, 1, CXLadderLinear, rz)

    @guppy
    @no_type_check
    def main() -> None:
        controls = qarray(1)
        qreg = qarray(1)
        h(controls[0])
        with control(controls):
            pauli_g(qreg, angle(theta))
        state_output("result_state", controls[0], qreg[0])
        discard_array(controls)
        discard_array(qreg)

    @guppy
    @no_type_check
    def main_ctrl_daggered() -> None:
        controls = qarray(1)
        qreg = qarray(1)
        h(controls[0])
        with control(controls):
            with dagger:
                pauli_g(qreg, angle(theta))
        state_output("result_state", controls[0], qreg[0])
        discard_array(controls)
        discard_array(qreg)

    expected_phase = np.exp(-1j * 0.5 * np.pi * theta)
    expected_state = np.array(
        [1.0, expected_phase, 0.0, 0.0], dtype=np.complex128
    ) / np.sqrt(2)
    assert_allclose_ignorephase(get_statevector(main, 2), expected_state)
    assert_allclose_ignorephase(
        get_statevector(main_ctrl_daggered, 2), expected_state.conj()
    )
