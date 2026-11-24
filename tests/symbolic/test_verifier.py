"""Tests of the symbolic verifier."""

from itertools import product
from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import array, control, dagger, nat
from guppylang.std.quantum import cx, discard, h, measure, qubit, reset, toffoli, x

from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.primitives.gate_decompositions.cnx.cnx import _rtof4
from guppyalgos.testing import get_unitary
from tests.symbolic.symint import xor
from tests.symbolic.verifier import verify


def and_into(a, b, t):
    """Flip ``t`` if ``a`` and ``b`` are set."""
    return a, b, xor(t, a & b)


@guppy
@no_type_check
def correct_circuit(a: qubit, b: qubit, t: qubit) -> None:
    """Flip ``t`` by the AND of ``a`` and ``b``."""
    toffoli(a, b, t)


@guppy
@no_type_check
def wrong_output(a: qubit, b: qubit, t: qubit) -> None:
    """Flip ``t`` by ``a`` alone."""
    cx(a, t)


@guppy
@no_type_check
def ancilla_left_in_one(a: qubit, b: qubit, t: qubit) -> None:
    """Discard an ancilla left in |1>, though it does not depend on the inputs."""
    q = qubit()
    x(q)
    discard(q)
    toffoli(a, b, t)


@guppy
@no_type_check
def reset_dirty_ancilla(a: qubit, b: qubit, t: qubit) -> None:
    """Reset an ancilla that still holds the AND, instead of uncomputing it."""
    q = qubit()
    toffoli(a, b, q)
    cx(q, t)
    reset(q)
    discard(q)


@guppy
@no_type_check
def unsupported_gate(a: qubit, b: qubit, t: qubit) -> None:
    """Apply two H gates, which cancel but are not reversible logic."""
    h(a)
    h(a)
    toffoli(a, b, t)


@guppy
@no_type_check
def phase_left(a: qubit, b: qubit, t: qubit) -> None:
    """Undo a relative-phase CCCX by three Toffolis, which leaves its phase."""
    d, e = qubit(), qubit()
    _rtof4(a, b, t, d)
    toffoli(a, b, e)
    toffoli(e, t, d)
    toffoli(a, b, e)
    discard(d)
    discard(e)
    toffoli(a, b, t)


@guppy
@no_type_check
def phase_on_other_controls(a: qubit, b: qubit, t: qubit) -> None:
    """Undo a relative-phase CCCX on swapped controls, which leaves its phase."""
    d = qubit()
    _rtof4(a, b, t, d)
    with dagger:
        _rtof4(a, t, b, d)
    discard(d)
    toffoli(a, b, t)


@guppy
@no_type_check
def rtof_on_dirty_target(a: qubit, b: qubit, t: qubit) -> None:
    """Apply a relative-phase CCCX onto ``t``, which does not start in |0>."""
    e = qubit()
    _rtof4(a, b, e, t)
    discard(e)


@guppy
@no_type_check
def and_on_dirty_target(a: qubit, b: qubit, t: qubit) -> None:
    """Compute the AND gadget onto ``t``, which does not start in |0>."""
    temp_and_compute(a, b, t)


@guppy
@no_type_check
def and_on_wrong_target(a: qubit, b: qubit, t: qubit) -> None:
    """Uncompute the AND gadget on a target that does not hold the AND."""
    q = qubit()
    temp_and_uncompute(a, b, q)
    discard(q)
    toffoli(a, b, t)


@guppy
@no_type_check
def unfixed_measurement(a: qubit, b: qubit, t: qubit) -> None:
    """Measure a copy of ``a``, whose value depends on the input."""
    q = qubit()
    cx(a, q)
    measure(q)
    toffoli(a, b, t)


def test_correct_circuit_passes():
    """The verifier accepts the circuit that its spec describes."""
    verify(correct_circuit, and_into)


def test_ancilla_count_is_checked():
    """The verifier rejects a circuit that does not use the ancillas it should."""
    with pytest.raises(AssertionError, match="the ancilla count is 0, not 1"):
        verify(correct_circuit, and_into, ancillas=1)


def test_namesake_of_a_gadget_runs_as_it_is():
    """Only the library's gadgets get a model, not functions sharing their name."""

    @guppy
    @no_type_check
    def _rtof4(a: qubit, b: qubit, c: qubit, d: qubit) -> None:
        toffoli(a, b, d)

    @guppy
    @no_type_check
    def circuit(a: qubit, b: qubit, t: qubit) -> None:
        q = qubit()
        _rtof4(a, b, q, t)
        discard(q)

    verify(circuit, and_into)


def test_custom_modifier_method_runs():
    """The verifier checks a custom modifier method, not tket's version of it."""

    @guppy.unitary
    class x_ignoring_control:
        @guppy
        @no_type_check
        def __call__(q: qubit) -> None:
            x(q)

        @guppy
        @no_type_check
        def controlled[n: nat](q: qubit, ctrls: array[qubit, n]) -> None:
            x(q)  # whatever the controls, unlike tket's control of __call__

    @guppy
    @no_type_check
    def circuit(c: qubit, q: qubit) -> None:
        with control(c):
            x_ignoring_control(q)

    verify(circuit, lambda c, q: (c, ~q))


WRONG_CIRCUITS = [  # (circuit, the error it raises, a part of its message)
    (wrong_output, AssertionError, "t bit 0 is wrong, counterexample: a=1, b=0"),
    (ancilla_left_in_one, AssertionError, "discarded while not in"),
    (reset_dirty_ancilla, AssertionError, "reset while it depends on the inputs"),
    (unsupported_gate, NotImplementedError, "tket.quantum.H"),
    (phase_left, AssertionError, "relative-phase CCCX is never uncomputed"),
    (phase_on_other_controls, AssertionError, "undone on other controls"),
    (rtof_on_dirty_target, AssertionError, "CCCX needs its target in"),
    (and_on_dirty_target, AssertionError, "AND gadget needs its target in"),
    (and_on_wrong_target, AssertionError, "AND gadget is uncomputed on a wrong"),
    (unfixed_measurement, AssertionError, "measured while its value is not fixed"),
]


@pytest.mark.parametrize(
    ("circuit", "error", "message"),
    WRONG_CIRCUITS,
    ids=[circuit.wrapped.name for circuit, _, _ in WRONG_CIRCUITS],
)
def test_wrong_circuit_fails(circuit, error, message):
    """The verifier rejects a wrong circuit, and says why."""
    with pytest.raises(error, match=message):
        verify(circuit, and_into)


def test_rtof4():
    """Test that ``_rtof4`` and its dagger do what sim.py's model assumes."""

    @guppy
    @no_type_check
    def rtof(q: array[qubit, 4]) -> None:
        _rtof4(q[0], q[1], q[2], q[3])

    @guppy
    @no_type_check
    def irtof(q: array[qubit, 4]) -> None:
        with dagger:
            _rtof4(q[0], q[1], q[2], q[3])

    u, u_inverse = get_unitary(rtof, 4), get_unitary(irtof, 4)
    assert np.allclose(u_inverse @ u, np.eye(16))
    phases = {}
    for a, b, c in product((0, 1), repeat=3):
        column = u[:, 8 * a + 4 * b + 2 * c]
        out = column[8 * a + 4 * b + 2 * c + (a & b & c)]
        assert np.isclose(abs(out), 1)
        assert np.isclose(out, phases.setdefault((a & b, c), out))
