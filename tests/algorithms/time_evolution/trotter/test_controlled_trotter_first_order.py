"""Tests for controlled first-order Trotterization."""

from __future__ import annotations

from typing import no_type_check

from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.builtins import array
from guppylang.std.debug import state_output
from guppylang.std.quantum import crz, discard_array, h, qubit
import numpy as np
import pytest
from pytest_lazy_fixtures import lf as lazy_fixture
import zixy.qubit.pauli as zqp
from scipy.linalg import expm

from guppyalgos.algorithms.time_evolution.trotter import cntrl_trotter_first_order
from guppyalgos.utils import qarray
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_statevector,
    get_unitary_projected,
)
from guppyalgos.utils import trotter_step_matrix


def cntrl_trotter_blocks(
    ham_op: zqp.RealTermSum,
    n_state_qubits: int,
    time_step: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return the ``00``, ``01``, ``10``, and ``11`` control blocks for one step."""
    # TODO: Replace repeated projected-unitary reconstruction with a helper that
    # extracts all control blocks from one full unitary once the register-aware
    # statevector/unitary utilities are improved.
    controlled_step = cntrl_trotter_first_order(ham_op, n_state_qubits)

    @guppy
    @no_type_check
    def main(
        ctrl_qreg: array[qubit, 1],
        state_qreg: array[qubit, n_state_qubits],
    ) -> None:
        controlled_step(ctrl_qreg[0], state_qreg, time_step)
        state_output("ctrl", ctrl_qreg)

    block_00 = get_unitary_projected(
        main, n_state_qubits, {"ctrl": [False]}, {"ctrl": [False]}
    )
    block_01 = get_unitary_projected(
        main, n_state_qubits, {"ctrl": [False]}, {"ctrl": [True]}
    )
    block_10 = get_unitary_projected(
        main, n_state_qubits, {"ctrl": [True]}, {"ctrl": [False]}
    )
    block_11 = get_unitary_projected(
        main, n_state_qubits, {"ctrl": [True]}, {"ctrl": [True]}
    )
    return block_00, block_01, block_10, block_11


def assert_cntrl_trotter_blocks_match(
    ham_op: zqp.RealTermSum,
    time_step: float,
) -> None:
    """Check the control blocks against the classical first-order Trotter step."""
    n_state_qubits = len(ham_op.qubits)

    block_00, block_01, block_10, block_11 = cntrl_trotter_blocks(
        ham_op, n_state_qubits, time_step
    )

    expected_u = trotter_step_matrix(ham_op, time_step, little_endian=True)
    identity_block = np.eye(2**n_state_qubits, dtype=np.complex128)
    zero_block = np.zeros_like(identity_block)

    assert_allclose_ignorephase(block_00, identity_block)
    np.testing.assert_allclose(block_01, zero_block, atol=1e-8)
    np.testing.assert_allclose(block_10, zero_block, atol=1e-8)
    assert_allclose_ignorephase(block_11, expected_u)


def cntrl_trotter_statevector(
    ham_op: zqp.RealTermSum,
    time_step: float,
) -> np.ndarray:
    """Return the two-qubit statevector after one controlled Trotter step."""
    controlled_step = cntrl_trotter_first_order(ham_op, 1)

    @guppy
    @no_type_check
    def main() -> None:
        ctrl_qreg = qarray(1)
        state_qreg = qarray(1)

        h(ctrl_qreg[0])
        controlled_step(ctrl_qreg[0], state_qreg, time_step)

        state_output("result_state", ctrl_qreg[0], state_qreg[0])
        discard_array(ctrl_qreg)
        discard_array(state_qreg)

    return get_statevector(main, 2)


def assert_cntrl_trotter_cntrl_phase(
    ham_op: zqp.RealTermSum, time_step: float, expected_phase: complex
) -> None:
    """Check that a one-qubit controlled Trotter step kicks back the right phase."""
    result_state = cntrl_trotter_statevector(ham_op, time_step)
    expected_state = np.array(
        [1.0, expected_phase, 0.0, 0.0], dtype=np.complex128
    ) / np.sqrt(2)

    # TODO: Once we have cleaner register-projection helpers, assert the kicked-back
    # control state directly without exposing the full two-qubit statevector here.
    assert_allclose_ignorephase(result_state, expected_state)
    np.testing.assert_allclose(
        result_state[1] / result_state[0], expected_phase, atol=1e-8
    )


@pytest.mark.parametrize(
    "ham_op",
    [
        lazy_fixture("ham_1q_posreal_2"),
        lazy_fixture("ham_2q_posreal_1"),
    ],
)
def test_cntrl_trotter_blocks(ham_op: zqp.RealTermSum) -> None:
    """Check ``00 ~= I``, ``01 = 10 = 0``, and ``11 ~=`` the classical Trotter step."""
    time_step = 0.3
    assert_cntrl_trotter_blocks_match(ham_op, time_step)


@pytest.mark.parametrize(
    "ham_op",
    [
        lazy_fixture("ham_2q_posreal_1"),
    ],
)
def test_cntrl_trotter_default_n_state_qubits(
    ham_op: zqp.RealTermSum,
) -> None:
    """Check omitted ``n_state_qubits`` defaults to ``len(ham_op.qubits)``."""
    n_state_qubits = len(ham_op.qubits)
    time_step = 0.3
    controlled_step = cntrl_trotter_first_order(ham_op)

    @guppy
    @no_type_check
    def main(
        ctrl_qreg: array[qubit, 1],
        state_qreg: array[qubit, n_state_qubits],
    ) -> None:
        controlled_step(ctrl_qreg[0], state_qreg, time_step)
        state_output("ctrl", ctrl_qreg)

    block_11 = get_unitary_projected(
        main, n_state_qubits, {"ctrl": [True]}, {"ctrl": [True]}
    )
    expected_u = trotter_step_matrix(ham_op, time_step, little_endian=True)
    assert_allclose_ignorephase(block_11, expected_u)


def test_cntrl_trotter_identity_phase() -> None:
    """Check that an identity term kicks back the expected phase onto the control."""
    ham_op = zqp.RealTermSum.from_str("(0.5, I0)")
    time_step = 0.7
    expected_phase = np.exp(-1j * 0.5 * np.pi * 0.5 * time_step)
    assert_cntrl_trotter_cntrl_phase(ham_op, time_step, expected_phase)


def test_cntrl_trotter_single_qubit_z() -> None:
    """Check the ``0.5 * Z`` toy model matches a direct ``crz`` reference exactly."""
    phi = 0.33
    time_step = -4 * phi
    ham_op = zqp.RealTermSum.from_str("(0.5, Z0)")
    expected_phase = np.exp(1j * np.pi * phi)
    assert_cntrl_trotter_cntrl_phase(ham_op, time_step, expected_phase)

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

    crz_state = get_statevector(crz_main, 2)
    expected_state = np.array(
        [1.0, expected_phase, 0.0, 0.0], dtype=np.complex128
    ) / np.sqrt(2)

    assert_allclose_ignorephase(crz_state, expected_state)
    np.testing.assert_allclose(crz_state[1] / crz_state[0], expected_phase, atol=1e-8)


def test_cntrl_trotter_identity_and_single_qubit_z() -> None:
    """Check identity and non-identity terms accumulate on the control branch."""
    phi = 0.2
    time_step = -4 * phi
    ham_op = zqp.RealTermSum.from_str("(0.25, I0), (0.5, Z0)")
    expected_phase = np.exp(1j * 1.5 * np.pi * phi)
    assert_cntrl_trotter_cntrl_phase(ham_op, time_step, expected_phase)


def test_cntrl_trotter_commuting_terms(
    ham_commuting_comp_basis_fixture: tuple[zqp.RealTermSum, float],
) -> None:
    """Check small computational-basis-diagonal commuting examples.

    The active controlled block is compared against the classical Trotter
    matrix.
    """
    ham_op, time_step = ham_commuting_comp_basis_fixture
    assert_cntrl_trotter_blocks_match(ham_op, time_step)


def test_cntrl_trotter_noncommuting_terms(
    ham_noncommuting_fixture: tuple[zqp.RealTermSum, float],
) -> None:
    """Check small noncommuting examples against the classical Trotter matrix."""
    ham_op, time_step = ham_noncommuting_fixture
    assert_cntrl_trotter_blocks_match(ham_op, time_step)


def test_commuting_trotter_step_matches_exact_exponential(
    ham_commuting_comp_basis_fixture: tuple[zqp.RealTermSum, float],
) -> None:
    """Check exactness for computational-basis-diagonal commuting examples."""
    ham_op, time_step = ham_commuting_comp_basis_fixture
    ham_mat = ham_op.to_sparse_matrix(True).toarray()
    exact_step = expm(-1j * (0.5 * np.pi * time_step) * ham_mat)
    trotter_step = trotter_step_matrix(ham_op, time_step, little_endian=True)

    np.testing.assert_allclose(trotter_step, exact_step, atol=1e-8)


def test_noncommuting_trotter_step_differs_from_exact_exponential(
    ham_noncommuting_fixture: tuple[zqp.RealTermSum, float],
) -> None:
    """Check that first-order Trotter differs from the exact step."""
    ham_op, time_step = ham_noncommuting_fixture
    ham_mat = ham_op.to_sparse_matrix(True).toarray()
    exact_step = expm(-1j * (0.5 * np.pi * time_step) * ham_mat)
    trotter_step = trotter_step_matrix(ham_op, time_step, little_endian=True)

    assert np.linalg.norm(trotter_step - exact_step) > 1e-6
