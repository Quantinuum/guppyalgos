"""Pauli Exponential tests."""

from __future__ import annotations
from guppylang import guppy

from guppylang.std.builtins import array
from guppylang.std.angles import angle
from guppylang.std.quantum import qubit, rz
import pytest

from guppyalgos.primitives.subroutines.ladders import CXLadderLinear, Ladder


import zixy.qubit.pauli as zqp

from guppylang.defs import GuppyFunctionDefinition

from guppyalgos.primitives.pauli.pauli_exp import pauli_exp

from guppyalgos.testing import get_unitary
from tests.primitives.pauli.pauli_exp.pauli_exp_helpers import pauli_exp_matrix
from guppyalgos.testing import assert_allclose_ignorephase
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


def test_pauli_exp_invalid_pauli() -> None:
    """Test that pauli exponential raises error for invalid pauli strings.

    The pauli exponential requires at least 1 non-identity Pauli operator. This
    test checks that a ValueError is raised for the all-identity case.
    """
    pauli_string = zqp.String.from_str("I0 I1 I2", 3)

    with pytest.raises(
        ValueError,
        match="Pauli exponential requires at least 1 non-identity Pauli operators",
    ):
        pauli_exp(pauli_string, 3, CXLadderLinear, rz)
