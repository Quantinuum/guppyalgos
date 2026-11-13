"""Tests for first-order Trotterization implementation."""

from guppyalgos.algorithms.time_evolution.trotter import (
    trotter_first_order,
    trotter_higher_order,
)

from guppylang import guppy
from guppylang.std.builtins import array
from guppyalgos.testing import get_unitary
from guppyalgos.testing import assert_allclose_ignorephase
from guppyalgos.utils import trotter_step_matrix
from guppylang.std.quantum import qubit
import pytest
from pytest_lazy_fixtures import lf as lazy_fixture
from guppyalgos.algorithms.time_evolution.trotter import ham_sim_trotter
from numpy.linalg import matrix_power
import numpy as np
from scipy.linalg import expm
import zixy.qubit.pauli as zqp


from typing import no_type_check


@pytest.mark.parametrize(
    "ham_op",
    [
        lazy_fixture("ham_2q_posreal_1"),  # reduced set of fixtures for faster testing
        lazy_fixture("ham_2q_posreal_2"),
        lazy_fixture("ham_2q_posreal_3"),
        lazy_fixture("ham_2q_posreal_4"),
    ],
)
def test_ham_sim_trotter_first_order(ham_op: zqp.RealTermSum) -> None:
    """Test full Hamiltonian simulation using first-order Trotterization.

    The test constructs a full Hamiltonian simulation function by applying multiple
    first-order Trotter steps for the provided Hamiltonian. It compares the resulting
    unitary against the expected unitary computed from the shared classical Trotter
    step helper and raised to the power of the number of Trotter steps.

    Note as of now, the implementation does not account for global phase factors,
    so the comparison is done with ``assert_allclose_ignorephase``.

    The tests run over a subset of the zixy RealTermSum Hamiltonian fixtures in
    conftest.py. Because the full set is too large for practical test times and is
    tested in test_trotter_first_order.py.

    Args:
        ham_op: The Hamiltonian operator to test, represented as a real Pauli term sum.

    """
    time_step = 0.1
    n_steps = 3
    n_state_qubits = len(ham_op.qubits)

    ham_trotter_step = trotter_first_order(ham_op, n_state_qubits)

    @guppy
    @no_type_check
    def main(state_qreg: array[qubit, n_state_qubits]) -> None:
        ham_sim_trotter(state_qreg, ham_trotter_step, n_steps, time_step)

    guppy_u = get_unitary(main, n_state_qubits)
    step_u = trotter_step_matrix(ham_op, time_step, little_endian=True)

    # Raise the single step unitary to the power of n_steps to get the expected
    # full simulation unitary
    expected_u = matrix_power(step_u, n_steps)

    assert_allclose_ignorephase(expected_u, guppy_u)


def test_ham_sim_trotter_higher_order_matches_exact_evolution() -> None:
    """Check that ham_sim_trotter composes a higher-order Trotter step."""
    ham_op = zqp.RealTermSum.from_str("(0.5, Z0), (0.1, X0)")
    n_state_qubits = 1
    n_steps = 4
    time_step = 0.05

    trotter_step = trotter_higher_order(ham_op, n_state_qubits, order=4)

    @guppy
    @no_type_check
    def main(state_qreg: array[qubit, n_state_qubits]) -> None:
        ham_sim_trotter(state_qreg, trotter_step, n_steps, time_step)

    actual = get_unitary(main, n_state_qubits)
    hamiltonian_matrix = ham_op.to_sparse_matrix(True).toarray()
    exact = expm(-0.5j * np.pi * (n_steps * time_step) * hamiltonian_matrix)

    assert_allclose_ignorephase(actual, exact, threshold=1e-6)
