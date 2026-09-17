"""Tests for first-order Trotterization implementation."""

from guppyalgos.algorithms.time_evolution.trotter import trotter_first_order
from guppylang import guppy
from guppylang.std.builtins import array
from guppyalgos.testing import get_unitary
from guppyalgos.testing import assert_allclose_ignorephase
from guppyalgos.utils import trotter_step_matrix
from guppylang.std.quantum import qubit
import pytest
from pytest_lazy_fixtures import lf as lazy_fixture
import zixy.qubit.pauli as zqp


from typing import no_type_check


@pytest.mark.parametrize(
    "ham_op",
    [
        lazy_fixture("op_hermitian_fixture"),
    ],
)
@pytest.mark.parametrize("provide_n_state_qubits", [True, False])
def test_trotter_step_first_order(
    ham_op: zqp.RealTermSum, provide_n_state_qubits: bool
) -> None:
    """Test single first-order Trotter step against expected unitary.

    The test constructs a first-order Trotter step for the provided Hamiltonian
    and compares the resulting unitary against the expected unitary computed
    using matrix exponentiation of the individual Pauli terms (trotter_step_matrix).

    Note as of now, the implementation does not account for global phase factors,
    so the comparison is done with ``assert_allclose_ignorephase``.

    The tests run over all the zixy RealTermSum Hamiltonian fixtures in
    conftest.py.

    Args:
        ham_op: The Hamiltonian operator to test, represented as a real Pauli term sum.
        provide_n_state_qubits: Is the number of qubits is passed to the trotter step.

    """
    n_state_qubits = len(ham_op.qubits)

    if provide_n_state_qubits:
        ham_trotter_step = trotter_first_order(ham_op, n_state_qubits)
    else:
        ham_trotter_step = trotter_first_order(ham_op)

    time_step = 0.3

    @guppy
    @no_type_check
    def main(state_qreg: array[qubit, n_state_qubits]) -> None:
        ham_trotter_step(state_qreg, time_step)

    guppy_u = get_unitary(main, n_state_qubits)
    expected_u = trotter_step_matrix(ham_op, time_step, little_endian=True)

    assert_allclose_ignorephase(expected_u, guppy_u)
