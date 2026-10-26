"""Modifier dispatch through generic callbacks and runtime walk powers."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang_internals.error import GuppyTypeInferenceError
from guppylang.std.builtins import array, Unitary
from guppylang.std.quantum import qubit, h, t, rz, angle
from pytket.circuit import Circuit

from guppyalgos.algorithms.time_evolution.trotter.ham_sim_trotter import ham_sim_trotter
from guppyalgos.testing import (
    assert_unitary_modifiers,
)


@guppy(unitary=True)
@no_type_check
def trotter_callback(qs: array[qubit, 1], step: float) -> None:
    """Noncommuting elementary step with a runtime classical parameter."""
    h(qs[0])
    rz(qs[0], angle(step))


@pytest.mark.parametrize("steps", [0, 1, 3])
def test_trotter_callback_modifiers(steps: int) -> None:
    """Check repeated noncommuting steps and their exact adjoints."""
    circuit = ham_sim_trotter(trotter_callback, steps, -0.3, 1)
    reference = np.linalg.matrix_power(Circuit(1).H(0).Rz(-0.3, 0).get_unitary(), steps)
    assert_unitary_modifiers(circuit, 1, expected_unitary=reference)


@pytest.mark.xfail(
    strict=True,
    raises=GuppyTypeInferenceError,
    reason="Guppy 1.1.1 cannot infer Unit callbacks over generic registers",
)
def test_generic_callback_protocol_inference() -> None:
    """Track the boundary blocking Hadamard-test and qubitized-walk migration."""

    @guppy(unitary=True)
    @no_type_check
    def generic[Regs](qs: Regs, op: Unitary[[Regs], None]) -> None:
        op(qs)

    @guppy(unitary=True)
    @no_type_check
    def operation(q: qubit) -> None:
        h(q)
        t(q)

    @guppy(unitary=True)
    @no_type_check
    def circuit(q: qubit) -> None:
        generic(q, operation)

    circuit.compile_function()
