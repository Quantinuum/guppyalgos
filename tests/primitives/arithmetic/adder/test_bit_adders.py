"""Tests for single-bit adders."""

from typing import no_type_check

import numpy as np
from guppylang.decorator import guppy
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard, qubit, x
from selene_sim import Quest
from guppyalgos.primitives.arithmetic.adder.bit_adders import (
    full_adder,
    full_adder_inverse,
    half_adder,
    half_adder_inverse,
)
from guppyalgos.testing import assert_allclose_ignorephase


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
    """Verify half_adder_inverse acts as the inverse."""

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
        half_adder_inverse(a, b, c_out)
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


def test_full_adder_inverse() -> None:
    """Verify full_adder_inverse does act as the inverse."""

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
        full_adder_inverse(cin, a, b, c_out)

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
