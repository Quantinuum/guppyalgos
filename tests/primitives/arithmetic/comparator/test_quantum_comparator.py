"""Tests for quantum comparators."""

from typing import no_type_check
from collections.abc import Callable

import pytest
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard, discard_array, qubit
from selene_sim import Quest
from guppyalgos.testing import project_state_onto_bitstring

import numpy as np

from guppyalgos.primitives.arithmetic.comparator import (
    comparator_ripple_cuccaro,
    comparator_vandaele,
    partial_comparator_ripple_cuccaro,
)
from guppyalgos.utils import apply_bitstring, int_to_bits, qarray


@pytest.mark.parametrize(
    ("n", "a", "b"),
    [
        (2, 0, 0),
        (2, 1, 2),
        (2, 3, 1),
        (3, 0, 7),
        (3, 2, 5),
        (3, 6, 1),
        (4, 3, 3),
        (4, 4, 11),
        (5, 5, 6),
        (5, 12, 3),
        (6, 5, 18),
        (7, 5, 100),
    ],
)
@pytest.mark.parametrize(
    ("make_comparator", "num_ancilla_fn"),
    [
        (lambda n: comparator_ripple_cuccaro, lambda n: 1),
        (comparator_vandaele, lambda n: 0),
    ],
    ids=["ripple_cuccaro", "vandaele"],
)
def test_compare(
    n: int,
    a: int,
    b: int,
    make_comparator: Callable[[int], GuppyFunctionDefinition],
    num_ancilla_fn: Callable[[int], int],
) -> None:
    """Test compare circuits against the classical comparison."""
    comparator = make_comparator(n)
    n_qubits = 2 * n + 1 + num_ancilla_fn(n)

    a_bits = int_to_bits(a, n)
    b_bits = int_to_bits(b, n)
    c = a < b

    @guppy
    @no_type_check
    def main() -> None:
        """Run the main test function."""
        a_reg = qarray(n)
        b_reg = qarray(n)
        comp = qubit()

        apply_bitstring(a_reg, a_bits)
        apply_bitstring(b_reg, b_bits)

        comparator(a_reg, b_reg, comp)

        state_output("a", a_reg)
        state_output("b", b_reg)
        state_output("comp", comp)

        discard_array(a_reg)
        discard_array(b_reg)
        discard(comp)

    statevector = main.emulator(n_qubits=n_qubits).with_seed(42).run()
    states = Quest.extract_states_dict(statevector.results[0].entries)

    comp_state = states["comp"].get_single_state()
    assert comp_state[int(c)] == 1

    a_state = project_state_onto_bitstring(states["a"], a_bits)
    assert np.allclose(a_state.probability, 1.0)

    b_state = project_state_onto_bitstring(states["b"], b_bits)
    assert np.allclose(b_state.probability, 1.0)


@pytest.mark.parametrize(
    ("n", "a", "b"),
    [
        (2, 0, 0),
        (2, 1, 2),
        (2, 3, 1),
        (3, 0, 7),
        (3, 2, 5),
        (3, 6, 1),
        (4, 3, 3),
        (4, 4, 11),
        (5, 5, 6),
        (5, 12, 3),
        (6, 5, 18),
        (7, 5, 100),
    ],
)
def test_partial_compare(n: int, a: int, b: int) -> None:
    """Test partial ripple carry compare circuit."""
    n_qubits = 2 * (n + 1) + 1

    a_bits = int_to_bits(a, n)
    b_bits = int_to_bits(b, n)
    c = a < b

    @guppy
    @no_type_check
    def main() -> None:
        """Run the main test function."""
        a_reg = qarray(n)
        b_reg = qarray(n)
        comp = qubit()

        apply_bitstring(a_reg, a_bits)
        apply_bitstring(b_reg, b_bits)

        ancilla = qubit()

        partial_comparator_ripple_cuccaro(a_reg, b_reg, comp, ancilla, False, True)
        state_output("comp", comp)

        partial_comparator_ripple_cuccaro(a_reg, b_reg, comp, ancilla, True, True)

        state_output("a_restored", a_reg)
        state_output("b_restored", b_reg)
        state_output("comp_uncomputed", comp)

        discard(ancilla)
        discard_array(a_reg)
        discard_array(b_reg)
        discard(comp)

    statevector = main.emulator(n_qubits=n_qubits).with_seed(42).run()
    states = Quest.extract_states_dict(statevector.results[0].entries)

    comp_state = states["comp"].get_single_state()
    assert comp_state[int(c)] == 1

    a_restored = project_state_onto_bitstring(states["a_restored"], a_bits)
    assert np.allclose(a_restored.probability, 1.0)

    b_restored = project_state_onto_bitstring(states["b_restored"], b_bits)
    assert np.allclose(b_restored.probability, 1.0)
