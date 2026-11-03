"""Phase primitive unitary modifier checks."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.quantum import angle, qubit

from guppyalgos.primitives.gate_decompositions.cnx.ccu_rel_phase import (
    ccu_phase_correction,
    ccz_rel_phase,
)
from guppyalgos.testing import assert_unitary_modifiers
from guppyalgos.utils.guppy.gates import apply_phase, t_state


@pytest.mark.parametrize("theta", [-0.75, 0.0, 0.33])
def test_apply_phase_modifiers(theta: float) -> None:
    """Check Rz and its modifiers against the intended phase convention."""

    @guppy(unitary=True)
    @no_type_check
    def operation(qs: array[qubit, 1]) -> None:
        apply_phase(qs[0], angle(theta))

    expected = np.diag(np.exp(np.pi * 1j * theta * np.array([-0.5, 0.5])))
    assert_unitary_modifiers(operation, 1, expected_unitary=expected)


def test_t_state_modifiers() -> None:
    """Check H followed by T and its modifiers on arbitrary inputs."""

    @guppy(unitary=True)
    @no_type_check
    def operation(qs: array[qubit, 1]) -> None:
        t_state(qs[0])

    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    expected = np.diag([1, np.exp(0.25j * np.pi)]) @ hadamard
    assert_unitary_modifiers(operation, 1, expected_unitary=expected)


def test_ccz_rel_phase_modifiers() -> None:
    """Check the relative-phase CCZ and its modifiers."""

    @guppy(unitary=True)
    @no_type_check
    def operation(qs: array[qubit, 3]) -> None:
        ccz_rel_phase(qs[0], qs[1], qs[2])

    expected = np.diag([1, 1, 1, 1, 1, 1, -1j, 1j])
    assert_unitary_modifiers(operation, 3, expected_unitary=expected)


def test_ccu_phase_correction_modifiers() -> None:
    """Check the CCU phase correction and its modifiers."""

    @guppy(unitary=True)
    @no_type_check
    def operation(qs: array[qubit, 2]) -> None:
        ccu_phase_correction(qs[0], qs[1])

    expected = np.diag(np.array([1, 1, 1, 1j]))
    assert_unitary_modifiers(operation, 2, expected_unitary=expected)
