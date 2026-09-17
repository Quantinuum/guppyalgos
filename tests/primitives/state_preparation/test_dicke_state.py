"""Tests for the Dicke state preparation."""

from __future__ import annotations

from typing import no_type_check

import numpy as np
import math
from itertools import combinations
import pytest
from guppylang import guppy
from guppylang.std.debug import state_result
from guppylang.std.quantum import discard_array

from guppyalgos.primitives.state_preparation.dicke_state import dicke_nk
from guppyalgos.utils import qarray
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_statevector,
)


def get_dicke_state(n, k):
    """Generate the Dicke state |D_k^(n)> as a numpy array."""
    if k < 0 or k > n:
        raise ValueError(f"k must be between 0 and {n}, got {k}")

    dim = 2**n
    state = np.zeros(dim, dtype=complex)

    ones_positions = combinations(range(n), k)
    norm = 1.0 / np.sqrt(math.comb(n, k))

    for positions in ones_positions:
        bits = ["0"] * n
        for pos in positions:
            bits[pos] = "1"
        binary_str = "".join(bits)

        index = int(binary_str, 2)
        state[index] = norm

    return state


@pytest.mark.parametrize("n_qubits", [4, 5, 6, 7, 8])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
@no_type_check
def test_dicke_state_statevector(n_qubits: int, k: int) -> None:
    """Test that dicke_nk produces the statevector for the given Dicke state."""

    @guppy
    def main() -> None:
        qs = qarray(n_qubits)
        dicke_nk(qs, k)
        state_result("result_state", qs)
        discard_array(qs)

    simulated = get_statevector(main, n_qubits)
    expected = get_dicke_state(n_qubits, k)
    assert_allclose_ignorephase(simulated, expected)
