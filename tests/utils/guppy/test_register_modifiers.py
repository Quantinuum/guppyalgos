"""Public modifier integration checks for the migrated circuits."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.quantum import qubit, h, t
from guppyalgos.testing import assert_unitary_modifiers

from guppyalgos.utils.guppy.gates import (
    transversal1,
    transversal1_start_end,
    transversal1_all,
    apply_bitstring,
)


@guppy(unitary=True)
def noncommuting(q: qubit) -> None:
    """Use a callback with noncommuting gates."""
    h(q)
    t(q)


@pytest.mark.parametrize("kind", ["ignore", "slice", "all", "bits", "frozen"])
def test_register_modifiers(kind: str) -> None:
    """Cover runtime predicates and a non-self-adjoint callback through public APIs."""
    if kind == "ignore":

        @guppy(unitary=True)
        @no_type_check
        def circuit(qs: array[qubit, 3]) -> None:
            transversal1(noncommuting, qs, array(nat(1), nat(1)))

    elif kind == "slice":

        @guppy(unitary=True)
        @no_type_check
        def circuit(qs: array[qubit, 3]) -> None:
            transversal1_start_end(noncommuting, qs, 1, 3)

    elif kind == "all":

        @guppy(unitary=True)
        @no_type_check
        def circuit(qs: array[qubit, 3]) -> None:
            transversal1_all(noncommuting, qs)

    elif kind == "bits":

        @guppy(unitary=True)
        @no_type_check
        def circuit(qs: array[qubit, 3]) -> None:
            apply_bitstring(qs, array(True, False, True))

    elif kind == "frozen":

        @guppy(unitary=True)
        @no_type_check
        def circuit(qs: array[qubit, 3]) -> None:
            apply_bitstring(qs, comptime([True, False, True]))

    one = np.diag([1, np.exp(0.25j * np.pi)]) @ (
        np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    )
    identity = np.eye(2)
    if kind == "ignore":
        expected = np.kron(np.kron(one, identity), one)
    elif kind == "slice":
        expected = np.kron(identity, np.kron(one, one))
    elif kind == "all":
        expected = np.kron(one, np.kron(one, one))
    else:
        x = np.array([[0, 1], [1, 0]])
        expected = np.kron(x, np.kron(identity, x))
    assert_unitary_modifiers(circuit, 3, expected_unitary=expected)
