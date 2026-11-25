"""Test CCU gate decompositions."""

import pytest
import numpy as np
from guppylang import guppy
from guppylang.std.quantum import qubit, array, angle
from guppylang.defs import GuppyFunctionDefinition
from guppyalgos.primitives.gate_decompositions.cnx.ccu import (
    ccry_cx,
    ccry_toffoli,
    ccx,
    ccy,
    ccz,
)
from guppyalgos.primitives.gate_decompositions.cnx.ccu_rel_phase import (
    ccu_phase_correction,
    ccx_rel_phase,
    ccz_rel_phase,
)
from guppyalgos.testing import assert_unitary_modifiers


from typing import no_type_check


@pytest.mark.parametrize("ccry_fun", [ccry_toffoli, ccry_cx])
@pytest.mark.parametrize("theta", [-0.3, 0.0, 0.25])
def test_ccry_modifiers(
    ccry_fun: GuppyFunctionDefinition[[qubit, qubit, qubit, angle], None],
    theta: float,
) -> None:
    """Check every modifier against an independent CCRy matrix, including phases."""

    @guppy(unitary=True)
    @no_type_check
    def apply_ccry(qs: array[qubit, 3]) -> None:
        ccry_fun(qs[0], qs[1], qs[2], angle(theta))

    c = np.cos(theta * np.pi / 2)
    s = np.sin(theta * np.pi / 2)
    # Basis order: |control1, control2, target>. Only |11> rotates the target.
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, c, -s],
            [0, 0, 0, 0, 0, 0, s, c],
        ],
        dtype=np.complex128,
    )

    # Added control uses an AND flag and a three-control CNX with one ancilla.
    assert_unitary_modifiers(
        apply_ccry,
        3,
        expected_unitary=expected,
        n_extra_qubits=2,
    )


def test_ccz_rel_phase_modifiers() -> None:
    """Check the relative-phase CCZ and its modifiers."""

    @guppy(unitary=True)
    @no_type_check
    def operation(qs: array[qubit, 3]) -> None:
        ccz_rel_phase(qs[0], qs[1], qs[2])

    expected = np.diag([1, 1, 1, 1, 1, 1, -1j, 1j])
    assert_unitary_modifiers(operation, 3, expected_unitary=expected)


def test_ccx_rel_phase_modifiers() -> None:
    """Check the relative-phase CCX and its modifiers."""

    @guppy(unitary=True)
    @no_type_check
    def operation(qs: array[qubit, 3]) -> None:
        ccx_rel_phase(qs[0], qs[1], qs[2])

    expected = np.eye(8, dtype=np.complex128)
    expected[6:, 6:] = [[0, -1j], [-1j, 0]]
    assert_unitary_modifiers(operation, 3, expected_unitary=expected)


def test_ccx_modifiers() -> None:
    """Check the CCX gate and its modifiers."""

    @guppy(unitary=True)
    @no_type_check
    def operation(qs: array[qubit, 3]) -> None:
        ccx(qs[0], qs[1], qs[2])

    expected = np.eye(8, dtype=np.complex128)
    expected[6:, 6:] = [[0, 1], [1, 0]]
    assert_unitary_modifiers(operation, 3, expected_unitary=expected)


def test_ccy_modifiers() -> None:
    """Check the CCY gate and its modifiers."""

    @guppy(unitary=True)
    @no_type_check
    def operation(qs: array[qubit, 3]) -> None:
        ccy(qs[0], qs[1], qs[2])

    expected = np.eye(8, dtype=np.complex128)
    expected[6:, 6:] = [[0, -1j], [1j, 0]]
    assert_unitary_modifiers(operation, 3, expected_unitary=expected)


def test_ccz_modifiers() -> None:
    """Check the CCZ gate and its modifiers."""

    @guppy(unitary=True)
    @no_type_check
    def operation(qs: array[qubit, 3]) -> None:
        ccz(qs[0], qs[1], qs[2])

    expected = np.diag([1, 1, 1, 1, 1, 1, 1, -1])
    assert_unitary_modifiers(operation, 3, expected_unitary=expected)


def test_ccu_phase_correction_modifiers() -> None:
    """Check the CCU phase correction and its modifiers."""

    @guppy(unitary=True)
    @no_type_check
    def operation(qs: array[qubit, 2]) -> None:
        ccu_phase_correction(qs[0], qs[1])

    expected = np.diag(np.array([1, 1, 1, 1j]))
    assert_unitary_modifiers(operation, 2, expected_unitary=expected)
