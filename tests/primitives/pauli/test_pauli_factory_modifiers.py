"""Check Pauli factories against independent matrices in public modifier modes."""

from typing import no_type_check

import numpy as np
import pytest
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, control
from guppylang.std.quantum import qubit

from guppyalgos.algorithms.select.pauli_select import _select_pauli_term
from guppyalgos.primitives.pauli import pauli_to_gate
from guppyalgos.primitives.pauli.pauli_exp import pauli_exp
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    assert_unitary_modifiers,
    get_unitary,
)
from tests.primitives.pauli.pauli_exp.pauli_exp_helpers import pauli_exp_matrix


def test_pauli_gate_modifiers() -> None:
    """Check all modes of a Pauli string containing X, Y, and Z."""
    pauli = zqp.String.from_str("X0 Y1 Z2", 3)
    expected = pauli.to_sparse_matrix(True).toarray()
    assert_unitary_modifiers(pauli_to_gate(pauli, 3), 3, expected_unitary=expected)


def test_phased_lcu_term_modifiers() -> None:
    """Preserve the coefficient phase across both levels of control."""
    pauli = zqp.String.from_str("Y0", 1)
    phase = 0.37
    gate = _select_pauli_term(pauli, 1, phase)

    @guppy.comptime(unitary=True)
    @no_type_check
    def circuit(qs: array[qubit, 2]) -> None:
        gate(qs[0], [qs[1]])

    expected = np.eye(4, dtype=complex)
    expected[2:, 2:] = (
        np.exp(1j * np.pi * phase) * pauli.to_sparse_matrix(True).toarray()
    )
    assert_unitary_modifiers(circuit, 2, expected_unitary=expected)


@pytest.mark.parametrize("controlled", [False, True])
def test_pauli_exp_modifiers(controlled: bool) -> None:
    """Check both exponential signatures, including a gap in sparse support."""
    pauli = zqp.String.from_str("Y0 X2", 3)
    theta = 0.37
    forward = pauli_exp(pauli, 3)
    expected = pauli_exp_matrix(pauli, 3, theta)

    if not controlled:

        @guppy(unitary=True)
        @no_type_check
        def circuit(qs: array[qubit, 3]) -> None:
            forward(qs, angle(theta))

        assert_unitary_modifiers(circuit, 3, expected_unitary=expected)
        return

    @guppy(unitary=True)
    @no_type_check
    def apply_controlled(flag: qubit, qs: array[qubit, 3]) -> None:
        with control(flag):
            forward(qs, angle(theta))

    @guppy.comptime(unitary=True)
    @no_type_check
    def controlled_circuit(qs: array[qubit, 4]) -> None:
        apply_controlled(qs[0], [qs[1], qs[2], qs[3]])

    controlled_expected = np.eye(16, dtype=complex)
    controlled_expected[8:, 8:] = expected
    assert_unitary_modifiers(
        controlled_circuit, 4, expected_unitary=controlled_expected
    )


def test_pauli_exp_two_controls() -> None:
    """Apply the exponential coherently only when both controls are active."""
    pauli = zqp.String.from_str("Y0 X1", 2)
    theta = -0.19
    forward = pauli_exp(pauli, 2)

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, 4]) -> None:
        targets = array(qs.take(2), qs.take(3))
        controls = array(qs.take(0), qs.take(1))
        with control(controls):
            forward(targets, angle(theta))
        qs.put(controls.take(0), 0)
        qs.put(controls.take(1), 1)
        qs.put(targets.take(0), 2)
        qs.put(targets.take(1), 3)
        controls.discard_all_taken()
        targets.discard_all_taken()

    expected = np.eye(16, dtype=complex)
    expected[12:, 12:] = pauli_exp_matrix(pauli, 2, theta)
    assert_allclose_ignorephase(get_unitary(circuit, 4), expected)
