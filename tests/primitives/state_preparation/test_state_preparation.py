"""Tests for state preparations.

It is sufficient to test the correctness of the prepared state.
We do not need to test the correctness of a unitary operation
as a state can be prepared by multiple different unitary operations.
"""

from guppyalgos.primitives.arithmetic.comparator import (
    comparator_ripple_cuccaro,
)

from itertools import product

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.quantum import discard_array
from selene_sim import Quest
from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx
from guppyalgos.primitives.state_preparation.uniform import uniform_state
from guppyalgos.primitives.state_preparation.ghz import ghz_state
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_statevector,
    switch_endianness,
)
from guppylang.std.debug import state_output
from guppyalgos.utils import qarray

from typing import no_type_check


@pytest.mark.parametrize("n_qubits", [5, 8, 11])
def test_uniform_state(n_qubits: int) -> None:
    """Test the uniform_state module for L is divisible by 2."""
    uniform_state_prep = uniform_state(2**n_qubits)

    @guppy
    @no_type_check
    def main() -> None:
        qs = qarray(n_qubits)
        uniform_state_prep(qs)
        state_output("result_state", qs)
        discard_array(qs)

    simulated_state = get_statevector(main, n_qubits)
    expected_state = np.array([1 / (2 ** (n_qubits / 2))] * (2**n_qubits))
    assert_allclose_ignorephase(simulated_state, expected_state)


@pytest.mark.parametrize(
    ("n_qubits", "is_log_depth"), list(product([5, 8, 11], [True, False]))
)
def test_ghz_state(n_qubits: int, is_log_depth: bool) -> None:
    """Test the GHZ state preparation."""

    @guppy
    @no_type_check
    def main() -> None:
        qs = qarray(n_qubits)
        ghz_state(qs)
        state_output("result_state", qs)
        discard_array(qs)

    simulated_state = get_statevector(main, n_qubits)
    expected_state = np.array(
        [
            1 / np.sqrt(2) if i == 0 or i == (2**n_qubits - 1) else 0
            for i in range(2**n_qubits)
        ]
    )
    assert_allclose_ignorephase(simulated_state, expected_state)


@pytest.mark.parametrize("L", [3, 5, 6, 10, 12, 15])
def test_uniform_non_div(L: int) -> None:
    """Test for the uniform_state module for L is non-divisible by 2."""
    n = int(np.ceil(np.log2(L)))
    comparator = comparator_ripple_cuccaro
    uniform = uniform_state(
        L,
        cnx,
        comparator,
    )

    @guppy
    @no_type_check
    def main() -> None:
        """Apply main."""
        qreg = qarray(n)
        uniform(qreg)
        state_output("prep", qreg)
        discard_array(qreg)

    statevector = main.emulator(n_qubits=2 * n + 5).with_seed(42).run()
    states = Quest.extract_states_dict(statevector.results[0].entries)
    work_state = switch_endianness(states["prep"].get_single_state())
    expected_state = np.array([0.0] * 2**n)
    expected_state[:L] = 1 / np.sqrt(L)

    assert np.allclose(
        expected_state,
        work_state,
        atol=1e-14,
    )
