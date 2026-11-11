"""Tests for single-bit adders."""

from typing import no_type_check

import numpy as np
from guppylang.decorator import guppy
from guppylang.std.builtins import control, dagger
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard, discard_array, h, qubit, x
from selene_sim import Quest
from guppyalgos.primitives.arithmetic.adder.bit_adders import (
    full_adder,
    half_adder,
)
from guppyalgos.testing import assert_allclose_ignorephase
from guppyalgos.testing import get_total_state_on_only_specified_registers
from guppyalgos.utils import qarray


def _full_adder_basis_output(
    cin_bit: bool,
    a_bit: bool,
    b_bit: bool,
    carry_bit: bool,
    *,
    daggered: bool,
) -> tuple[bool, bool, bool, bool]:
    """Return the basis-state output of full_adder or its inverse."""
    b_for_carry = b_bit ^ cin_bit ^ a_bit if daggered else b_bit
    carry_bit ^= (
        (cin_bit and a_bit) or (cin_bit and b_for_carry) or (a_bit and b_for_carry)
    )
    b_bit ^= cin_bit ^ a_bit
    return cin_bit, a_bit, b_bit, carry_bit


def test_half_adder_truth_table() -> None:
    """Verify half_adder implements sum = a xor b and carry = a and b."""

    @guppy
    @no_type_check
    def main(a_bit: bool, b_bit: bool) -> None:
        a = qubit()
        b = qubit()
        c_out = qubit()

        if a_bit:
            x(a)
        if b_bit:
            x(b)

        half_adder(a, b, c_out)

        state_output("sum", b)
        state_output("carry", c_out)
        discard(a)
        discard(b)
        discard(c_out)

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(3)
    for a_bit, b_bit, expected_sum, expected_carry in [
        (False, False, False, False),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, True),
    ]:
        res = emulator.run(a_bit=a_bit, b_bit=b_bit)
        states = Quest.extract_states_dict(res.results[0].entries)
        zero_state = np.array([1, 0], dtype=np.complex128)
        one_state = np.array([0, 1], dtype=np.complex128)
        if expected_sum:
            assert_allclose_ignorephase(states["sum"].get_single_state(), one_state)
        else:
            assert_allclose_ignorephase(states["sum"].get_single_state(), zero_state)
        if expected_carry:
            assert_allclose_ignorephase(states["carry"].get_single_state(), one_state)
        else:
            assert_allclose_ignorephase(states["carry"].get_single_state(), zero_state)


def test_full_adder_truth_table() -> None:
    """Verify full_adder computes a+b+cin with sum in b and carry in c_out."""

    @guppy
    @no_type_check
    def main(a_bit: bool, b_bit: bool, cin_bit: bool) -> None:
        cin = qubit()
        a = qubit()
        b = qubit()
        c_out = qubit()

        if cin_bit:
            x(cin)
        if a_bit:
            x(a)
        if b_bit:
            x(b)

        full_adder(cin, a, b, c_out)

        state_output("sum", b)
        state_output("carry", c_out)
        discard(a)
        discard(b)
        discard(cin)
        discard(c_out)

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(4)
    for cin_bit, a_bit, b_bit, expected_sum, expected_carry in [
        (False, False, False, False, False),
        (False, False, True, True, False),
        (False, True, False, True, False),
        (False, True, True, False, True),
        (True, False, False, True, False),
        (True, False, True, False, True),
        (True, True, False, False, True),
        (True, True, True, True, True),
    ]:
        res = emulator.run(a_bit=a_bit, b_bit=b_bit, cin_bit=cin_bit)
        states = Quest.extract_states_dict(res.results[0].entries)
        zero_state = np.array([1, 0], dtype=np.complex128)
        one_state = np.array([0, 1], dtype=np.complex128)
        if expected_sum:
            assert_allclose_ignorephase(states["sum"].get_single_state(), one_state)
        else:
            assert_allclose_ignorephase(states["sum"].get_single_state(), zero_state)
        if expected_carry:
            assert_allclose_ignorephase(states["carry"].get_single_state(), one_state)
        else:
            assert_allclose_ignorephase(states["carry"].get_single_state(), zero_state)


def test_half_adder_inverse() -> None:
    """Verify daggered half_adder acts as the inverse."""

    @guppy
    @no_type_check
    def main(a_bit: bool, b_bit: bool) -> None:
        a = qubit()
        b = qubit()
        c_out = qubit()

        if a_bit:
            x(a)
        if b_bit:
            x(b)

        half_adder(a, b, c_out)
        with dagger:
            half_adder(a, b, c_out)
        if a_bit:
            x(a)
        if b_bit:
            x(b)

        state_output("a", a)
        state_output("b", b)
        state_output("carry", c_out)
        discard(a)
        discard(b)
        discard(c_out)

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(3)
    for a_bit, b_bit in [(False, False), (False, True), (True, False), (True, True)]:
        res = emulator.run(a_bit=a_bit, b_bit=b_bit)
        states = Quest.extract_states_dict(res.results[0].entries)
        zero_state = np.array([1, 0], dtype=np.complex128)
        for state in states.values():
            assert_allclose_ignorephase(state.get_single_state(), zero_state)


def test_full_adder_dagger() -> None:
    """Verify daggered full_adder acts as the inverse."""

    @guppy
    @no_type_check
    def main(cin_bit: bool, a_bit: bool, b_bit: bool) -> None:
        cin = qubit()
        a = qubit()
        b = qubit()
        c_out = qubit()

        if cin_bit:
            x(cin)
        if a_bit:
            x(a)
        if b_bit:
            x(b)

        full_adder(cin, a, b, c_out)
        with dagger:
            full_adder(cin, a, b, c_out)

        if cin_bit:
            x(cin)
        if a_bit:
            x(a)
        if b_bit:
            x(b)

        state_output("sum", b)
        state_output("carry", c_out)
        discard(a)
        discard(b)
        discard(cin)
        discard(c_out)

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(4)
    for cin_bit, a_bit, b_bit in [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ]:
        res = emulator.run(cin_bit=cin_bit, a_bit=a_bit, b_bit=b_bit)
        states = Quest.extract_states_dict(res.results[0].entries)
        zero_state = np.array([1, 0], dtype=np.complex128)
        for state in states.values():
            assert_allclose_ignorephase(state.get_single_state(), zero_state)


def test_half_adder_controlled_modifier_with_superposition_controls() -> None:
    """Check controlled half_adder preserves coherent two-control branches."""

    @guppy
    @no_type_check
    def main() -> None:
        controls = qarray(2)
        h(controls[0])
        h(controls[1])
        a = qubit()
        b = qubit()
        c_out = qubit()
        x(a)
        x(b)

        with control(controls):
            half_adder(a, b, c_out)

        state_output("controls", controls)
        state_output("a", a)
        state_output("b", b)
        state_output("carry", c_out)
        discard_array(controls)
        discard(a)
        discard(b)
        discard(c_out)

    states = Quest.extract_states_dict(main.emulator(5).run().results[0].entries)
    total_state, _ = get_total_state_on_only_specified_registers(
        states, ["controls", "a", "b", "carry"]
    )
    expected_state = np.zeros(2**5, dtype=np.complex128)
    for control_value in range(4):
        # Qubit positions are controls[0], controls[1], a, b, and carry.
        b_value = 0 if control_value == 3 else 1
        carry_value = 1 if control_value == 3 else 0
        index = control_value + (1 << 2) + (b_value << 3) + (carry_value << 4)
        expected_state[index] = 0.5

    assert_allclose_ignorephase(total_state.state, expected_state)


def test_half_adder_controlled_daggered_modifier_with_superposition_controls() -> None:
    """Check controlled daggered half_adder preserves coherent control branches."""

    @guppy
    @no_type_check
    def main() -> None:
        controls = qarray(2)
        h(controls[0])
        h(controls[1])
        a = qubit()
        b = qubit()
        c_out = qubit()
        x(a)
        x(c_out)

        with control(controls), dagger:
            half_adder(a, b, c_out)

        state_output("controls", controls)
        state_output("a", a)
        state_output("b", b)
        state_output("carry", c_out)
        discard_array(controls)
        discard(a)
        discard(b)
        discard(c_out)

    states = Quest.extract_states_dict(main.emulator(5).run().results[0].entries)
    total_state, _ = get_total_state_on_only_specified_registers(
        states, ["controls", "a", "b", "carry"]
    )
    expected_state = np.zeros(2**5, dtype=np.complex128)
    for control_value in range(4):
        # Qubit positions are controls[0], controls[1], a, b, and carry.
        b_value = 1 if control_value == 3 else 0
        carry_value = 0 if control_value == 3 else 1
        index = control_value + (1 << 2) + (b_value << 3) + (carry_value << 4)
        expected_state[index] = 0.5

    assert_allclose_ignorephase(total_state.state, expected_state)


def test_full_adder_controlled_modifier_with_superposition_controls() -> None:
    """Check controlled full_adder preserves coherent two-control branches."""

    @guppy
    @no_type_check
    def main(cin_bit: bool, a_bit: bool, b_bit: bool, carry_bit: bool) -> None:
        controls = qarray(2)
        h(controls[0])
        h(controls[1])
        cin = qubit()
        a = qubit()
        b = qubit()
        c_out = qubit()
        if cin_bit:
            x(cin)
        if a_bit:
            x(a)
        if b_bit:
            x(b)
        if carry_bit:
            x(c_out)

        with control(controls):
            full_adder(cin, a, b, c_out)

        state_output("controls", controls)
        state_output("cin", cin)
        state_output("a", a)
        state_output("b", b)
        state_output("carry", c_out)
        discard_array(controls)
        discard(cin)
        discard(a)
        discard(b)
        discard(c_out)

    emulator = main.emulator(6)
    for data_value in range(16):
        cin_bit = bool(data_value & 1)
        a_bit = bool(data_value & 2)
        b_bit = bool(data_value & 4)
        carry_bit = bool(data_value & 8)
        res = emulator.run(
            cin_bit=cin_bit,
            a_bit=a_bit,
            b_bit=b_bit,
            carry_bit=carry_bit,
        )
        states = Quest.extract_states_dict(res.results[0].entries)
        total_state, _ = get_total_state_on_only_specified_registers(
            states, ["controls", "cin", "a", "b", "carry"]
        )
        expected_state = np.zeros(2**6, dtype=np.complex128)
        for control_value in range(4):
            if control_value == 3:
                output_bits = _full_adder_basis_output(
                    cin_bit, a_bit, b_bit, carry_bit, daggered=False
                )
            else:
                output_bits = cin_bit, a_bit, b_bit, carry_bit
            output_cin, output_a, output_b, output_carry = output_bits
            index = (
                control_value
                + (int(output_cin) << 2)
                + (int(output_a) << 3)
                + (int(output_b) << 4)
                + (int(output_carry) << 5)
            )
            expected_state[index] = 0.5

        assert_allclose_ignorephase(total_state.state, expected_state)


def test_full_adder_controlled_daggered_modifier_with_superposition_controls() -> None:
    """Check controlled daggered full_adder preserves coherent control branches."""

    @guppy
    @no_type_check
    def main(cin_bit: bool, a_bit: bool, b_bit: bool, carry_bit: bool) -> None:
        controls = qarray(2)
        h(controls[0])
        h(controls[1])
        cin = qubit()
        a = qubit()
        b = qubit()
        c_out = qubit()
        if cin_bit:
            x(cin)
        if a_bit:
            x(a)
        if b_bit:
            x(b)
        if carry_bit:
            x(c_out)

        with control(controls), dagger:
            full_adder(cin, a, b, c_out)

        state_output("controls", controls)
        state_output("cin", cin)
        state_output("a", a)
        state_output("b", b)
        state_output("carry", c_out)
        discard_array(controls)
        discard(cin)
        discard(a)
        discard(b)
        discard(c_out)

    emulator = main.emulator(6)
    for data_value in range(16):
        cin_bit = bool(data_value & 1)
        a_bit = bool(data_value & 2)
        b_bit = bool(data_value & 4)
        carry_bit = bool(data_value & 8)
        res = emulator.run(
            cin_bit=cin_bit,
            a_bit=a_bit,
            b_bit=b_bit,
            carry_bit=carry_bit,
        )
        states = Quest.extract_states_dict(res.results[0].entries)
        total_state, _ = get_total_state_on_only_specified_registers(
            states, ["controls", "cin", "a", "b", "carry"]
        )
        expected_state = np.zeros(2**6, dtype=np.complex128)
        for control_value in range(4):
            if control_value == 3:
                output_bits = _full_adder_basis_output(
                    cin_bit, a_bit, b_bit, carry_bit, daggered=True
                )
            else:
                output_bits = cin_bit, a_bit, b_bit, carry_bit
            output_cin, output_a, output_b, output_carry = output_bits
            index = (
                control_value
                + (int(output_cin) << 2)
                + (int(output_a) << 3)
                + (int(output_b) << 4)
                + (int(output_carry) << 5)
            )
            expected_state[index] = 0.5

        assert_allclose_ignorephase(total_state.state, expected_state)
