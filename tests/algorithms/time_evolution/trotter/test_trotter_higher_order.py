"""Tests for recursive higher-order Suzuki--Trotter formulas."""

from __future__ import annotations

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, h, qubit
import numpy as np
import pytest
from scipy.linalg import expm
import zixy.qubit.pauli as zqp

from guppyalgos.algorithms.time_evolution.trotter import (
    cntrl_trotter_higher_order,
    suzuki_sequence,
    trotter_higher_order,
)
from guppyalgos.utils import qarray
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_statevector,
    get_unitary,
    get_unitary_projected,
)


def _sequence_matrix(
    ham_op: zqp.RealTermSum,
    time_step: float,
    order: int,
) -> np.ndarray:
    terms: list[zqp.RealTerm] = list(  # ty: ignore[invalid-assignment]
        ham_op.to_terms()
    )
    terms = [term for term in terms if not term.string.is_identity()]
    unitary = np.eye(2 ** len(ham_op.qubits), dtype=np.complex128)

    for term_index, time_factor in suzuki_sequence(len(terms), order):
        term_matrix = terms[term_index].to_sparse_matrix(True).toarray()
        unitary = expm(-0.5j * np.pi * time_step * time_factor * term_matrix) @ unitary

    return unitary


def _higher_order_unitary(
    ham_op: zqp.RealTermSum, order: int, time_step: float
) -> np.ndarray:
    """Return a one-qubit higher-order Trotter unitary for a fixed time step."""
    trotter_step = trotter_higher_order(ham_op, 1, order)

    @guppy
    @no_type_check
    def main(state_qreg: array[qubit, 1]) -> None:
        trotter_step(state_qreg, time_step)

    return get_unitary(main, 1)


@pytest.mark.parametrize("order", [2, 4])
def test_higher_order_trotter_matches_suzuki_formula(order: int) -> None:
    """Check generated circuits against the corresponding matrix recursion."""
    ham_op = zqp.RealTermSum.from_str("(0.5, Z0), (0.1, X0)")
    time_step = 0.3
    n_state_qubits = 1
    trotter_step = trotter_higher_order(ham_op, n_state_qubits, order)

    @guppy
    @no_type_check
    def main(state_qreg: array[qubit, n_state_qubits]) -> None:
        trotter_step(state_qreg, time_step)

    actual = get_unitary(main, n_state_qubits)
    expected = _sequence_matrix(ham_op, time_step, order)

    assert_allclose_ignorephase(actual, expected)


@pytest.mark.parametrize(
    ("order", "expected_local_order"),
    [(2, 3), (4, 5), (6, 7)],
)
def test_higher_order_trotter_converges_to_exact_unitary(
    order: int, expected_local_order: int
) -> None:
    """Check local error scaling against the exact matrix exponential."""
    ham_op = zqp.RealTermSum.from_str("(0.5, Z0), (0.1, X0)")
    hamiltonian_matrix = ham_op.to_sparse_matrix(True).toarray()
    time_steps = [0.2, 0.1]
    errors = []

    for time_step in time_steps:
        actual = _higher_order_unitary(ham_op, order, time_step)
        exact = expm(-0.5j * np.pi * time_step * hamiltonian_matrix)

        phase = np.vdot(exact, actual)
        phase /= abs(phase)
        errors.append(np.linalg.norm(actual - phase * exact, ord="fro"))

    observed_ratio = errors[0] / errors[1]
    assert observed_ratio > 2**expected_local_order / 4


def test_fourth_order_recursion_uses_suzuki_coefficient() -> None:
    """Check the five recursively scaled second-order stages."""
    second_order = suzuki_sequence(2, 2)
    fourth_order = suzuki_sequence(2, 4)
    p_2 = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
    expected_scales = [p_2, p_2, 1.0 - 4.0 * p_2, p_2, p_2]

    stage_length = len(second_order)
    assert len(fourth_order) == 5 * stage_length
    for stage, scale in enumerate(expected_scales):
        expected = [
            (term_index, scale * time_factor)
            for term_index, time_factor in second_order
        ]
        start = stage * stage_length
        assert fourth_order[start : start + stage_length] == expected


@pytest.mark.parametrize("order", [0, 1, 3, 5, 2.0, True])
def test_higher_order_trotter_rejects_invalid_order(order: object) -> None:
    """Only even integer orders greater than or equal to two are supported."""
    ham_op = zqp.RealTermSum.from_str("(0.5, Z0)")

    with pytest.raises(ValueError, match="even integer"):
        trotter_higher_order(ham_op, 1, order)


def test_higher_order_trotter_omits_identity_only_hamiltonian() -> None:
    """An uncontrolled identity Hamiltonian produces a no-op up to global phase."""
    ham_op = zqp.RealTermSum.from_str("(0.5, I0)")
    trotter_step = trotter_higher_order(ham_op, 1, 4)

    @guppy
    @no_type_check
    def main(state_qreg: array[qubit, 1]) -> None:
        trotter_step(state_qreg, 0.3)

    actual = get_unitary(main, 1)
    np.testing.assert_allclose(actual, np.eye(2), atol=1e-8)


def test_cntrl_higher_order_trotter_blocks_match_formula() -> None:
    """Check inactive, off-diagonal, and active blocks of a fourth-order step."""
    ham_op = zqp.RealTermSum.from_str("(0.5, Z0), (0.1, X0)")
    time_step = 0.3
    order = 4
    controlled_step = cntrl_trotter_higher_order(ham_op, 1, order)

    @guppy
    @no_type_check
    def main(control_qreg: array[qubit, 1], state_qreg: array[qubit, 1]) -> None:
        controlled_step(control_qreg[0], state_qreg, time_step)
        state_output("control", control_qreg)

    block_00 = get_unitary_projected(
        main, 1, {"control": [False]}, {"control": [False]}
    )
    block_01 = get_unitary_projected(main, 1, {"control": [False]}, {"control": [True]})
    block_10 = get_unitary_projected(main, 1, {"control": [True]}, {"control": [False]})
    block_11 = get_unitary_projected(main, 1, {"control": [True]}, {"control": [True]})

    identity = np.eye(2, dtype=np.complex128)
    zero = np.zeros((2, 2), dtype=np.complex128)
    expected_active = _sequence_matrix(ham_op, time_step, order)
    assert_allclose_ignorephase(block_00, identity)
    np.testing.assert_allclose(block_01, zero, atol=1e-8)
    np.testing.assert_allclose(block_10, zero, atol=1e-8)
    assert_allclose_ignorephase(block_11, expected_active)


def test_cntrl_higher_order_trotter_retains_identity_phase() -> None:
    """Check identity terms kick back their phase onto the control branch."""
    coefficient = 0.5
    time_step = 0.7
    ham_op = zqp.RealTermSum.from_str(f"({coefficient}, I0)")
    controlled_step = cntrl_trotter_higher_order(ham_op, 1, 4)

    @guppy
    @no_type_check
    def main() -> None:
        control_qreg = qarray(1)
        state_qreg = qarray(1)
        h(control_qreg[0])
        controlled_step(control_qreg[0], state_qreg, time_step)
        state_output("result_state", control_qreg[0], state_qreg[0])
        discard_array(control_qreg)
        discard_array(state_qreg)

    actual = get_statevector(main, 2)
    expected_phase = np.exp(-0.5j * np.pi * coefficient * time_step)
    expected = np.array([1.0, expected_phase, 0.0, 0.0]) / np.sqrt(2)
    assert_allclose_ignorephase(actual, expected)
    np.testing.assert_allclose(actual[1] / actual[0], expected_phase, atol=1e-8)


@pytest.mark.parametrize("order", [1, 3, True])
def test_cntrl_higher_order_trotter_rejects_invalid_order(order: object) -> None:
    """The controlled factory enforces the same even-order contract."""
    ham_op = zqp.RealTermSum.from_str("(0.5, Z0)")

    with pytest.raises(ValueError, match="even integer"):
        cntrl_trotter_higher_order(ham_op, 1, order)
