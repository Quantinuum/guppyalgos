"""Tests for adders."""

from typing import no_type_check
from collections.abc import Callable

import pytest
import numpy as np

from guppylang.decorator import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import array, control, dagger, output, nat
from guppylang.std.debug import state_output
from guppylang.std.quantum import (
    collect_measurements,
    discard,
    discard_array,
    measure_array,
    qubit,
    ry,
    rz,
)
from selene_sim import Quest

from guppyalgos.primitives.arithmetic import (
    adder_ripple_cuccaro_carry_out,
    adder_ripple_cuccaro_mod,
    adder_ripple_gidney_carry_out,
    adder_ripple_gidney_mod,
)
from guppyalgos.utils import apply_bitstring, int_to_bits, qarray
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_total_state_on_only_specified_registers,
    project_state_onto_bitstring,
)


ADDITION_CASES = (
    pytest.param(1, ((0, 0), (0, 1), (1, 1)), id="n1"),
    pytest.param(2, ((3, 1), (1, 2), (3, 3)), id="n2"),
    pytest.param(3, ((3, 4), (5, 2), (7, 7)), id="n3"),
    pytest.param(4, ((3, 7), (9, 4), (15, 15)), id="n4"),
    pytest.param(5, ((5, 12), (18, 6), (31, 31)), id="n5"),
    pytest.param(6, ((5, 32), (48, 6)), id="n6"),
)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("n_controls", [0, 1, 2, 3])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize(
    ("adder", "with_carry"),
    [
        (adder_ripple_cuccaro_mod, False),
        (adder_ripple_gidney_mod, False),
        (adder_ripple_cuccaro_carry_out, True),
        (adder_ripple_gidney_carry_out, True),
    ],
)
def test_adder_modifiers_superposition[n: nat, n_controls: nat](
    n: int,
    n_controls: int,
    inverse: bool,
    adder: GuppyFunctionDefinition,
    with_carry: bool,
) -> None:
    """Check modifier phases and multiple controls using clean carry promises."""
    carry_angle = 0.0 if with_carry else 0.37
    if with_carry:

        @guppy(unitary=True)
        @no_type_check
        def apply(a_reg: array[qubit, n], b_reg: array[qubit, n], carry: qubit) -> None:
            adder(a_reg, b_reg, carry)

    else:

        @guppy(unitary=True)
        @no_type_check
        def apply(a_reg: array[qubit, n], b_reg: array[qubit, n], carry: qubit) -> None:
            adder(a_reg, b_reg)

    if inverse:

        @guppy(unitary=True)
        @no_type_check
        def directed(
            a_reg: array[qubit, n], b_reg: array[qubit, n], carry: qubit
        ) -> None:
            with dagger:
                apply(a_reg, b_reg, carry)

    else:
        directed = apply

    if n_controls:

        @guppy
        @no_type_check
        def modified(
            a_reg: array[qubit, n],
            b_reg: array[qubit, n],
            carry: qubit,
            controls: array[qubit, n_controls],
        ) -> None:
            if with_carry and inverse:
                with control(controls):
                    apply(a_reg, b_reg, carry)
            with control(controls):
                directed(a_reg, b_reg, carry)

    else:

        @guppy
        @no_type_check
        def modified(
            a_reg: array[qubit, n],
            b_reg: array[qubit, n],
            carry: qubit,
            controls: array[qubit, n_controls],
        ) -> None:
            if with_carry and inverse:
                apply(a_reg, b_reg, carry)
            directed(a_reg, b_reg, carry)

    @guppy
    @no_type_check
    def main() -> None:
        a_reg = qarray(n)
        b_reg = qarray(n)
        carry = qubit()
        controls = qarray(n_controls)
        for index in range(n):
            ry(a_reg[index], angle(0.13 + 0.07 * index))
            rz(a_reg[index], angle(0.11))
            ry(b_reg[index], angle(0.29 + 0.05 * index))
            rz(b_reg[index], angle(0.11))
        ry(carry, angle(carry_angle))
        rz(carry, angle(0.11))
        for index in range(n_controls):
            ry(controls[index], angle(0.41 + 0.03 * index))
            rz(controls[index], angle(0.11))

        modified(a_reg, b_reg, carry, controls)

        state_output("a", a_reg)
        state_output("b", b_reg)
        state_output("carry", carry)
        if n_controls > 0:
            state_output("controls", controls)
        discard_array(a_reg)
        discard_array(b_reg)
        discard(carry)
        discard_array(controls)

    angles = (
        [0.13 + 0.07 * index for index in range(n)]
        + [0.29 + 0.05 * index for index in range(n)]
        + [carry_angle]
        + [0.41 + 0.03 * index for index in range(n_controls)]
    )
    initial = np.array([1.0 + 0.0j])
    for angle_value in angles:
        amplitudes = np.array(
            [
                np.cos(np.pi * angle_value / 2) * np.exp(-0.055j * np.pi),
                np.sin(np.pi * angle_value / 2) * np.exp(0.055j * np.pi),
            ]
        )
        initial = np.kron(amplitudes, initial)

    expected = np.zeros_like(initial)
    mask = (1 << n) - 1
    for index, amplitude in enumerate(initial):
        a_value = index & mask
        b_value = (index >> n) & mask
        carry_value = (index >> (2 * n)) & 1
        controls_value = index >> (2 * n + 1)
        if controls_value == (1 << n_controls) - 1 and not (with_carry and inverse):
            total = b_value - a_value if inverse else b_value + a_value
            if with_carry:
                carry_value ^= int(total < 0 or total >= 1 << n)
            b_value = total & mask
        target = (
            a_value
            | (b_value << n)
            | (carry_value << (2 * n))
            | (controls_value << (2 * n + 1))
        )
        expected[target] = amplitude

    emulator = main.emulator(n_qubits=3 * n + n_controls + 5)
    for _ in range(3):
        result = emulator.run()
        states = Quest.extract_states_dict(result.results[0].entries)
        registers = ["a", "b", "carry"]
        if n_controls:
            registers.append("controls")
        actual, _ = get_total_state_on_only_specified_registers(states, registers)
        assert_allclose_ignorephase(actual.state, expected)


@pytest.mark.parametrize(("n", "cases"), ADDITION_CASES)
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


@pytest.mark.parametrize(("n", "cases"), ADDITION_CASES)
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


@pytest.mark.parametrize(("n", "cases"), ADDITION_CASES)
@pytest.mark.parametrize(
    ("adder", "num_ancilla_fn"),
    [
        (
            adder_ripple_cuccaro_carry_out,
            lambda n: 1,
        ),
        (
            adder_ripple_gidney_carry_out,
            lambda n: n - 1,
        ),
    ],
)
def test_addition_carry_out_dagger[n: nat](
    n: int,
    cases: list[tuple[int, int]],
    adder: GuppyFunctionDefinition[[array[qubit, n], array[qubit, n], qubit], None],
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
        with dagger:
            adder(a_reg, b_reg, carry_out)

        apply_bitstring(a_reg, _a_bit_array)
        apply_bitstring(b_reg, _b_bit_array)

        state_output("a_reg", a_reg)
        state_output("b_reg", b_reg)

        state_output("carry_out", carry_out)
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


@pytest.mark.parametrize(("n", "cases"), ADDITION_CASES)
@pytest.mark.parametrize(
    ("adder", "num_ancilla_fn"),
    [
        (adder_ripple_cuccaro_mod, lambda n: 1),
        (adder_ripple_gidney_mod, lambda n: n - 1),
    ],
)
def test_addition_mod_dagger[n: nat](
    n: int,
    cases: list[tuple[int, int]],
    adder: GuppyFunctionDefinition[[array[qubit, n], array[qubit, n]], None],
    num_ancilla_fn: Callable[[int], int],
) -> None:
    """Test ripple carry dagger inverts correctly."""
    n_qubits = 2 * n + 1 * num_ancilla_fn(n)

    @guppy
    @no_type_check
    def main(_a_bit_array: array[bool, n], _b_bit_array: array[bool, n]) -> None:
        """Run the main test function."""
        a_reg = qarray(n)
        b_reg = qarray(n)
        apply_bitstring(a_reg, _a_bit_array)
        apply_bitstring(b_reg, _b_bit_array)

        adder(a_reg, b_reg)
        with dagger:
            adder(a_reg, b_reg)

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
