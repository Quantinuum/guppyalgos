"""Tests for adders."""

from typing import no_type_check
from collections.abc import Callable

import pytest
import numpy as np

from guppylang.decorator import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, output, nat
from guppylang.std.debug import state_output
from guppylang.std.quantum import (
    collect_measurements,
    discard,
    discard_array,
    measure_array,
    qubit,
)
from selene_sim import Quest

from guppyalgos.primitives.arithmetic import (
    adder_ripple_cuccaro_carry_out,
    adder_ripple_cuccaro_carry_out_dagger,
    adder_ripple_cuccaro_mod,
    adder_ripple_cuccaro_mod_dagger,
    adder_ripple_gidney_carry_out,
    adder_ripple_gidney_carry_out_dagger,
    adder_ripple_gidney_mod,
    adder_ripple_gidney_mod_dagger,
)
from guppyalgos.utils import apply_bitstring, int_to_bits, qarray
from guppyalgos.testing import (
    get_total_state_on_only_specified_registers,
    project_state_onto_bitstring,
)


@pytest.mark.parametrize(
    ("n", "cases"),
    [
        (2, [(3, 1), (1, 2), (3, 3)]),
        (3, [(3, 4), (5, 2), (7, 7)]),
        (4, [(3, 7), (9, 4), (15, 15)]),
        (5, [(5, 12), (18, 6), (31, 31)]),
        (6, [(5, 32), (48, 6)]),
    ],
)
@pytest.mark.parametrize(
    ("adder", "num_ancilla_fn"),
    [
        (adder_ripple_cuccaro_carry_out, lambda n: 1),
        (adder_ripple_gidney_carry_out, lambda n: n - 1),
    ],
)
def test_addition_carry_out[n: nat](
    n: int,
    cases: list[tuple[int, int]],
    adder: GuppyFunctionDefinition[[array[qubit, n], array[qubit, n], qubit], None],
    num_ancilla_fn: Callable[[int], int],
) -> None:
    """Test ripple carry addition circuit."""
    n_qubits = 2 * n + 1 + num_ancilla_fn(n)

    @guppy
    @no_type_check
    def main(_a_bit_array: array[bool, n], _b_bit_array: array[bool, n]) -> None:
        """Run the main test function."""
        a_reg = qarray(n)
        b_reg = qarray(n)
        apply_bitstring(a_reg, _a_bit_array)
        apply_bitstring(b_reg, _b_bit_array)
        carry_out = qubit()

        adder(a_reg, b_reg, carry_out)

        state_output("carry_out", carry_out)
        state_output("a_reg", a_reg)
        state_output("b_reg", b_reg)

        discard(carry_out)
        discard_array(a_reg)

        output("b_meas", collect_measurements(measure_array(b_reg)))

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=n_qubits)
    for a, b in cases:
        a_bits = int_to_bits(a, n)

        b_bits = int_to_bits(b, n)

        a_plus_b = a + b

        res = emulator.run(_a_bit_array=a_bits, _b_bit_array=b_bits)
        assert res.results[0].as_dict()["b_meas"] == int_to_bits((a + b) % 2**n, n)

        states = Quest.extract_states_dict(res.results[0].entries)

        total_state, _ = get_total_state_on_only_specified_registers(
            states, ["b_reg", "carry_out"]
        )
        total_state = total_state.state
        assert total_state[a_plus_b] == 1

        a_proj = project_state_onto_bitstring(states["a_reg"], int_to_bits(a, n))
        assert np.allclose(a_proj.probability, 1.0)


@pytest.mark.parametrize(
    ("n", "cases"),
    [
        (2, [(3, 1), (1, 2), (3, 3)]),
        (3, [(3, 4), (5, 2), (7, 7)]),
        (4, [(3, 7), (9, 4), (15, 15)]),
        (5, [(5, 12), (18, 6), (31, 31)]),
        (6, [(5, 32), (48, 6)]),
    ],
)
@pytest.mark.parametrize(
    ("adder", "num_ancilla_fn"),
    [
        (adder_ripple_cuccaro_mod, lambda n: 1),
        (adder_ripple_gidney_mod, lambda n: n - 1),
    ],
)
def test_addition_mod[n: nat](
    n: int,
    cases: list[tuple[int, int]],
    adder: GuppyFunctionDefinition[[array[qubit, n], array[qubit, n]], None],
    num_ancilla_fn: Callable[[int], int],
) -> None:
    """Test ripple carry modular addition circuit."""
    n_qubits = 2 * n + num_ancilla_fn(n)

    @guppy
    @no_type_check
    def main(_a_bit_array: array[bool, n], _b_bit_array: array[bool, n]) -> None:
        """Run the main test function."""
        a_reg = qarray(n)
        b_reg = qarray(n)
        apply_bitstring(a_reg, _a_bit_array)
        apply_bitstring(b_reg, _b_bit_array)
        carry_out = qubit()
        discard(carry_out)

        adder(a_reg, b_reg)

        state_output("a_reg", a_reg)
        state_output("b_reg", b_reg)

        output("b_meas", collect_measurements(measure_array(b_reg)))

        discard_array(a_reg)

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=n_qubits)
    for a, b in cases:
        a_bits = int_to_bits(a, n)

        b_bits = int_to_bits(b, n)

        res = emulator.run(_a_bit_array=a_bits, _b_bit_array=b_bits)
        assert res.results[0].as_dict()["b_meas"] == int_to_bits((a + b) % 2**n, n)

        states = Quest.extract_states_dict(res.results[0].entries)

        total_state, _ = get_total_state_on_only_specified_registers(states, ["b_reg"])
        total_state = total_state.state

        a_plus_b_mod = (a + b) % 2**n
        assert total_state[a_plus_b_mod] == 1

        a_proj = project_state_onto_bitstring(states["a_reg"], int_to_bits(a, n))
        assert np.allclose(a_proj.probability, 1.0)


@pytest.mark.parametrize(
    ("n", "cases"),
    [
        (2, [(3, 1), (1, 2), (3, 3)]),
        (3, [(3, 4), (5, 2), (7, 7)]),
        (4, [(3, 7), (9, 4), (15, 15)]),
        (5, [(5, 12), (18, 6), (31, 31)]),
        (6, [(5, 32), (48, 6)]),
    ],
)
@pytest.mark.parametrize(
    ("adder", "adder_dagger", "num_ancilla_fn"),
    [
        (
            adder_ripple_cuccaro_carry_out,
            adder_ripple_cuccaro_carry_out_dagger,
            lambda n: 1,
        ),
        (
            adder_ripple_gidney_carry_out,
            adder_ripple_gidney_carry_out_dagger,
            lambda n: n - 1,
        ),
    ],
)
def test_addition_carry_out_dagger[n: nat](
    n: int,
    cases: list[tuple[int, int]],
    adder: GuppyFunctionDefinition[[array[qubit, n], array[qubit, n], qubit], None],
    adder_dagger: GuppyFunctionDefinition[
        [array[qubit, n], array[qubit, n], qubit], None
    ],
    num_ancilla_fn: Callable[[int], int],
) -> None:
    """Test ripple carry dagger inverts correctly."""
    n_qubits = 2 * n + 1 + num_ancilla_fn(n)

    @guppy
    @no_type_check
    def main(_a_bit_array: array[bool, n], _b_bit_array: array[bool, n]) -> None:
        """Run the main test function."""
        a_reg = qarray(n)
        b_reg = qarray(n)
        apply_bitstring(a_reg, _a_bit_array)
        apply_bitstring(b_reg, _b_bit_array)
        carry_out = qubit()

        adder(a_reg, b_reg, carry_out)
        adder_dagger(a_reg, b_reg, carry_out)

        apply_bitstring(a_reg, _a_bit_array)
        apply_bitstring(b_reg, _b_bit_array)

        state_output("a_reg", a_reg)
        state_output("b_reg", b_reg)

        # carry_out was measurement-uncomputed, so its post-measurement
        # value is irrelevant
        discard(carry_out)

        discard_array(a_reg)
        discard_array(b_reg)

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=n_qubits).with_seed(42)
    for a, b in cases:
        a_bits = int_to_bits(a, n)

        b_bits = int_to_bits(b, n)

        statevector = emulator.run(_a_bit_array=a_bits, _b_bit_array=b_bits)
        states = Quest.extract_states_dict(statevector.results[0].entries)
        for state in states.values():
            assert state.get_single_state()[0] == 1


@pytest.mark.parametrize(
    ("n", "cases"),
    [
        (2, [(3, 1), (1, 2), (3, 3)]),
        (3, [(3, 4), (5, 2), (7, 7)]),
        (4, [(3, 7), (9, 4), (15, 15)]),
        (5, [(5, 12), (18, 6), (31, 31)]),
        (6, [(5, 32), (48, 6)]),
    ],
)
@pytest.mark.parametrize(
    ("adder", "adder_dagger", "num_ancilla_fn"),
    [
        (adder_ripple_cuccaro_mod, adder_ripple_cuccaro_mod_dagger, lambda n: 1),
        (adder_ripple_gidney_mod, adder_ripple_gidney_mod_dagger, lambda n: n - 1),
    ],
)
def test_addition_mod_dagger[n: nat](
    n: int,
    cases: list[tuple[int, int]],
    adder: GuppyFunctionDefinition[[array[qubit, n], array[qubit, n], qubit], None],
    adder_dagger: GuppyFunctionDefinition[
        [array[qubit, n], array[qubit, n], qubit], None
    ],
    num_ancilla_fn: Callable[[int], int],
) -> None:
    """Test ripple carry dagger inverts correctly."""
    n_qubits = 2 * n + 2 * num_ancilla_fn(n)

    @guppy
    @no_type_check
    def main(_a_bit_array: array[bool, n], _b_bit_array: array[bool, n]) -> None:
        """Run the main test function."""
        a_reg = qarray(n)
        b_reg = qarray(n)
        apply_bitstring(a_reg, _a_bit_array)
        apply_bitstring(b_reg, _b_bit_array)
        carry_out = qubit()

        adder(a_reg, b_reg)
        adder_dagger(a_reg, b_reg)

        discard(carry_out)

        apply_bitstring(a_reg, _a_bit_array)
        apply_bitstring(b_reg, _b_bit_array)

        state_output("a_reg", a_reg)
        state_output("b_reg", b_reg)

        discard_array(a_reg)
        discard_array(b_reg)

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=n_qubits).with_seed(42)
    for a, b in cases:
        a_bits = int_to_bits(a, n)

        b_bits = int_to_bits(b, n)

        statevector = emulator.run(_a_bit_array=a_bits, _b_bit_array=b_bits)
        states = Quest.extract_states_dict(statevector.results[0].entries)
        for state in states.values():
            assert state.get_single_state()[0] == 1
