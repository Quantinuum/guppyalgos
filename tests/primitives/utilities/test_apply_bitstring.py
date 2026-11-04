"""Unitary modifier checks for both bitstring overloads."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import array, comptime, control
from guppylang.std.debug import state_output
from guppylang.std.platform import barrier
from guppylang.std.quantum import (
    discard_array,
    h,
    qubit,
)
from selene_sim import Quest

from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx
from guppyalgos.primitives.utilities.gates import apply_bitstring, transversal
from guppyalgos.testing import assert_unitary_modifiers
from guppyalgos.utils import qarray


@pytest.mark.parametrize("kind", ["bits", "frozen"])
def test_bitstring_modifiers(kind: str) -> None:
    """Check array and frozen-array bit flips through the public overload.

    `assert_unitary_modifiers` only checks single bit control.
    """
    if kind == "bits":

        @guppy(unitary=True)
        @no_type_check
        def circuit(qs: array[qubit, 3]) -> None:
            apply_bitstring(qs, array(True, False, True))

    else:

        @guppy(unitary=True)
        @no_type_check
        def circuit(qs: array[qubit, 3]) -> None:
            apply_bitstring(qs, comptime([True, False, True]))

    x = np.array([[0, 1], [1, 0]])
    expected = np.kron(x, np.kron(np.eye(2), x))
    assert_unitary_modifiers(circuit, 3, expected_unitary=expected)


@pytest.mark.parametrize("kind", ["bits", "frozen"])
def test_apply_bitstring_multicontrol(kind: str) -> None:
    """Check array and frozen-array bit flips with multicontrol."""
    if kind == "bits":

        @guppy
        @no_type_check
        def circuit(controls: array[qubit, 3], qs: array[qubit, 3]) -> None:
            with control(controls):
                apply_bitstring(qs, array(True, False, True))

    else:

        @guppy
        @no_type_check
        def circuit(controls: array[qubit, 3], qs: array[qubit, 3]) -> None:
            with control(controls):
                apply_bitstring(qs, comptime([True, False, True]))

    @guppy
    @no_type_check
    def with_fanout() -> None:
        qs = qarray(3)

        controls = qarray(3)
        transversal(h, controls)
        circuit(controls, qs)
        # invert flipped bits using individual cnxs
        cnx(controls, qs[0])
        cnx(controls, qs[2])
        transversal(h, controls)

        barrier(controls, qs)
        state_output("out", qs)
        discard_array(qs)
        discard_array(controls)

    n_ancilla = 2
    res = with_fanout.emulator(2 * 3 + n_ancilla).run()
    states = Quest.extract_states_dict(res.results[0].entries)
    assert abs(states["out"].state[0]) == pytest.approx(1)
