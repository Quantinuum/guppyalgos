"""Phase primitive correctness and public modifier compilation."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import array, control, dagger
from guppylang.std.quantum import angle, qubit

from guppyalgos.primitives.gate_decompositions.cnx.ccu_rel_phase import (
    ccu_phase_correction,
    ccz_rel_phase,
)
from guppyalgos.testing import assert_allclose_ignorephase, get_unitary
from guppyalgos.utils.guppy.gates import apply_phase, t_state


@pytest.mark.parametrize("theta", [-0.75, 0.0, 0.33])
def test_apply_phase(theta: float) -> None:
    """Preserve the Rz phase convention, rather than changing to a phase gate."""

    @guppy
    @no_type_check
    def operation(qs: array[qubit, 1]) -> None:
        apply_phase(qs[0], angle(theta))

    expected = np.diag(np.exp(np.pi * 1j * theta * np.array([-0.5, 0.5])))
    assert_allclose_ignorephase(get_unitary(operation, 1), expected)


def test_phase_circuits() -> None:
    """Check full-domain matrices independently of generated modifiers."""

    @guppy
    @no_type_check
    def prepare(qs: array[qubit, 1]) -> None:
        t_state(qs[0])

    @guppy
    @no_type_check
    def relative(qs: array[qubit, 3]) -> None:
        ccz_rel_phase(qs[0], qs[1], qs[2])

    @guppy
    @no_type_check
    def correction(qs: array[qubit, 2]) -> None:
        ccu_phase_correction(qs[0], qs[1])

    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    expected_t = np.diag([1, np.exp(0.25j * np.pi)]) @ hadamard
    assert_allclose_ignorephase(get_unitary(prepare, 1), expected_t)
    assert_allclose_ignorephase(
        get_unitary(relative, 3), np.diag([1, 1, 1, 1, 1, 1, -1j, 1j])
    )
    assert_allclose_ignorephase(get_unitary(correction, 2), np.diag([1, 1, 1, 1j]))


def test_phase_modifier_compilation() -> None:
    """Compile all public modes, including two external controls and an angle."""

    @guppy
    @no_type_check
    def program(qs: array[qubit, 3], controls: array[qubit, 2], theta: angle) -> None:
        apply_phase(qs[0], theta)
        t_state(qs[1])
        ccz_rel_phase(qs[0], qs[1], qs[2])
        ccu_phase_correction(qs[0], qs[1])
        with dagger:
            apply_phase(qs[0], theta)
            t_state(qs[1])
            ccz_rel_phase(qs[0], qs[1], qs[2])
            ccu_phase_correction(qs[0], qs[1])
        with control(controls):
            apply_phase(qs[0], theta)
            t_state(qs[1])
            ccz_rel_phase(qs[0], qs[1], qs[2])
            ccu_phase_correction(qs[0], qs[1])
        with control(controls), dagger:
            apply_phase(qs[0], theta)
            t_state(qs[1])
            ccz_rel_phase(qs[0], qs[1], qs[2])
            ccu_phase_correction(qs[0], qs[1])

    # Guppy 1.1.1's default optimizer panics when a runtime-angle operation and
    # its adjoint cancel in this mixed-mode circuit. Minimal optimization still
    # lowers the generated modifiers; keep this integration check explicit.
    program.with_minimal_opt().compile_function()

    @guppy
    @no_type_check
    def combined(qs: array[qubit, 3], controls: array[qubit, 2], theta: angle) -> None:
        with control(controls), dagger:
            apply_phase(qs[0], theta)
            t_state(qs[1])
            ccz_rel_phase(qs[0], qs[1], qs[2])
            ccu_phase_correction(qs[0], qs[1])

    combined.compile_function()
