"""Tests for single-bit adders."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang.decorator import guppy
from guppylang.std.builtins import array, control, dagger
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard, qubit, x
from selene_sim import Quest
from guppyalgos.primitives.arithmetic.adder.bit_adders import (
    full_adder,
    half_adder,
)
from guppyalgos.testing import (
    Endianness,
    assert_allclose_ignorephase,
    assert_unitary_modifiers,
)


@pytest.mark.parametrize("extra_control", [False, True])
def test_half_adder_modifiers(extra_control: bool) -> None:
    """Check every half-adder modifier against its complete truth table."""

    @guppy(unitary=True)
    @no_type_check
    def circuit(register: array[qubit, 3]) -> None:
        half_adder(register[0], register[1], register[2])

    expected = np.zeros((8, 8), dtype=np.complex128)
    for index in range(8):
        addend = index & 1
        target = (index >> 1) & 1
        carry = (index >> 2) & 1
        output_index = (
            addend | ((target ^ addend) << 1) | ((carry ^ (addend & target)) << 2)
        )
        expected[output_index, index] = 1

    width = 3
    if extra_control:

        @guppy(unitary=True)
        @no_type_check
        def circuit(register: array[qubit, 4]) -> None:
            with control(register[0]):
                half_adder(register[1], register[2], register[3])

        expected = np.kron(np.eye(8), np.diag([1, 0])) + np.kron(
            expected, np.diag([0, 1])
        )
        width = 4
    assert_unitary_modifiers(
        circuit,
        width,
        expected_unitary=expected,
        endianness=Endianness.LITTLE,
        n_extra_qubits=3,
    )


@pytest.mark.parametrize("extra_control", [False, True])
def test_full_adder_modifiers(extra_control: bool) -> None:
    """Check every full-adder modifier against its complete truth table."""

    @guppy(unitary=True)
    @no_type_check
    def circuit(register: array[qubit, 4]) -> None:
        full_adder(register[0], register[1], register[2], register[3])

    expected = np.zeros((16, 16), dtype=np.complex128)
    for index in range(16):
        carry_in = index & 1
        addend = (index >> 1) & 1
        target = (index >> 2) & 1
        carry_out = (index >> 3) & 1
        total = carry_in + addend + target
        output_index = (
            carry_in
            | (addend << 1)
            | ((total & 1) << 2)
            | ((carry_out ^ (total >> 1)) << 3)
        )
        expected[output_index, index] = 1

    width = 4
    if extra_control:

        @guppy(unitary=True)
        @no_type_check
        def circuit(register: array[qubit, 5]) -> None:
            with control(register[0]):
                full_adder(register[1], register[2], register[3], register[4])

        expected = np.kron(np.eye(16), np.diag([1, 0])) + np.kron(
            expected, np.diag([0, 1])
        )
        width = 5
    assert_unitary_modifiers(
        circuit,
        width,
        expected_unitary=expected,
        endianness=Endianness.LITTLE,
        n_extra_qubits=3,
    )


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


def test_half_adder_dagger() -> None:
    """Verify the dagger modifier inverts the half adder."""

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
    """Verify the dagger modifier inverts the full adder."""

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
