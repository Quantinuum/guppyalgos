"""Unitary modifier checks for both bitstring overloads."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import array, comptime
from guppylang.std.quantum import qubit

from guppyalgos.testing import assert_unitary_modifiers
from guppyalgos.utils.guppy.gates import apply_bitstring


@pytest.mark.parametrize("kind", ["bits", "frozen"])
def test_bitstring_modifiers(kind: str) -> None:
    """Check array and frozen-array bit flips through the public overload."""
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
