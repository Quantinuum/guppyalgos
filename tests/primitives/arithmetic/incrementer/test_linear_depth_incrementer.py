"""Tests for the linear-depth Gidney-style incrementer."""

from collections import Counter
from typing import no_type_check

import pytest
from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.quantum import measure, measure_array, qubit, x, collect_measurements
from guppyalgos.utils import qarray
from guppyalgos.primitives.arithmetic.incrementer.linear_depth_incrementer import (
    cntrl_linear_depth_decrementer,
    cntrl_linear_depth_incrementer,
    linear_depth_decrementer,
    linear_depth_incrementer,
)


def _little_endian_bits(value: int, num_bits: int) -> list[bool]:
    return [bool((value >> i) & 1) for i in range(num_bits)]


def _output_string(value: int, num_bits: int) -> str:
    bits = _little_endian_bits(value, num_bits)
    return "".join("1" if bit else "0" for bit in bits)


@pytest.mark.symbolic_alternative
def test_linear_depth_incrementer_two_bits() -> None:
    """Increment a 2-bit little-endian register."""

    @guppy
    @no_type_check
    def main(bits: array[bool, 2]) -> None:
        q = qarray(2)

        for i in range(2):
            if bits[i]:
                x(q[i])

        linear_depth_incrementer(q)
        output("out", collect_measurements(measure_array(q)))

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=2).with_seed(7).with_shots(10)
    for value in [0, 1, 2, 3]:
        input_bits = _little_endian_bits(value, 2)
        expected = _output_string((value + 1) % 4, 2)

        emulator_result = emulator.run(bits=input_bits)
        assert emulator_result.collated_counts() == Counter({(("out", expected),): 10})


@pytest.mark.symbolic_alternative
def test_linear_depth_incrementer_five_bits() -> None:
    """Increment a 5-bit little-endian register."""

    @guppy
    @no_type_check
    def main(bits: array[bool, 5]) -> None:
        q = qarray(5)

        if bits[0]:
            x(q[0])
        if bits[1]:
            x(q[1])
        if bits[2]:
            x(q[2])
        if bits[3]:
            x(q[3])
        if bits[4]:
            x(q[4])

        linear_depth_incrementer(q)
        output("out", collect_measurements(measure_array(q)))

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=8).with_seed(11).with_shots(10)
    for value in [0, 1, 13, 30, 31]:
        input_bits = _little_endian_bits(value, 5)
        expected = _output_string((value + 1) % 32, 5)

        emulator_result = emulator.run(bits=input_bits)
        assert emulator_result.collated_counts() == Counter({(("out", expected),): 10})


def test_linear_depth_incrementer_bounded_width() -> None:
    """Only the low bits needed by `max_value + 1` participate in the carry chain."""

    @guppy
    @no_type_check
    def main(bits: array[bool, 5]) -> None:
        q = qarray(5)

        for i in range(5):
            if bits[i]:
                x(q[i])

        linear_depth_incrementer(q, 14)
        output("out", collect_measurements(measure_array(q)))

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=7).with_seed(17).with_shots(10)
    for value in [0, 1, 2, 3, 14]:
        input_bits = _little_endian_bits(value, 5)
        expected = _output_string(value + 1, 5)

        emulator_result = emulator.run(bits=input_bits)
        assert emulator_result.collated_counts() == Counter({(("out", expected),): 10})


@pytest.mark.symbolic_alternative
def test_cntrl_linear_depth_incrementer() -> None:
    """Increment only when the control qubit is set."""

    @guppy
    @no_type_check
    def main(bits: array[bool, 5], control_on: bool) -> None:
        control = qubit()
        q = qarray(5)

        if control_on:
            x(control)
        if bits[0]:
            x(q[0])
        if bits[1]:
            x(q[1])
        if bits[2]:
            x(q[2])
        if bits[3]:
            x(q[3])
        if bits[4]:
            x(q[4])

        cntrl_linear_depth_incrementer(control, q)
        output("ctrl", measure(control).read())
        output("out", collect_measurements(measure_array(q)))

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=10).with_seed(13).with_shots(10)
    for control_on in [False, True]:
        for value in [0, 1, 13, 30, 31]:
            input_bits = _little_endian_bits(value, 5)
            expected_value = (value + 1) % 32 if control_on else value
            expected = _output_string(expected_value, 5)

            emulator_result = emulator.run(bits=input_bits, control_on=control_on)
            expected_ctrl = "1" if control_on else "0"
            assert emulator_result.collated_counts() == Counter(
                {(("ctrl", expected_ctrl), ("out", expected)): 10}
            )


def test_cntrl_linear_depth_incrementer_bounded_width() -> None:
    """Controlled incrementer respects the same inclusive bound optimization."""

    @guppy
    @no_type_check
    def main(bits: array[bool, 5], control_on: bool) -> None:
        control = qubit()
        q = qarray(5)

        if control_on:
            x(control)
        for i in range(5):
            if bits[i]:
                x(q[i])

        cntrl_linear_depth_incrementer(control, q, 14)
        output("ctrl", measure(control).read())
        output("out", collect_measurements(measure_array(q)))

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=9).with_seed(19).with_shots(10)
    for control_on in [False, True]:
        for value in [0, 1, 2, 3, 14]:
            input_bits = _little_endian_bits(value, 5)
            expected_value = value + 1 if control_on else value
            expected = _output_string(expected_value, 5)

            emulator_result = emulator.run(bits=input_bits, control_on=control_on)
            expected_ctrl = "1" if control_on else "0"
            assert emulator_result.collated_counts() == Counter(
                {(("ctrl", expected_ctrl), ("out", expected)): 10}
            )


@pytest.mark.symbolic_alternative
def test_linear_depth_decrementer_two_bits() -> None:
    """Decrement a 2-bit little-endian register."""

    @guppy
    @no_type_check
    def main(bits: array[bool, 2]) -> None:
        q = qarray(2)

        for i in range(2):
            if bits[i]:
                x(q[i])

        linear_depth_decrementer(q)
        output("out", collect_measurements(measure_array(q)))

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=2).with_seed(23).with_shots(10)
    for value in [0, 1, 2, 3]:
        input_bits = _little_endian_bits(value, 2)
        expected = _output_string((value - 1) % 4, 2)

        emulator_result = emulator.run(bits=input_bits)
        assert emulator_result.collated_counts() == Counter({(("out", expected),): 10})


@pytest.mark.symbolic_alternative
def test_linear_depth_decrementer_five_bits() -> None:
    """Decrement a 5-bit little-endian register."""

    @guppy
    @no_type_check
    def main(bits: array[bool, 5]) -> None:
        q = qarray(5)

        for i in range(5):
            if bits[i]:
                x(q[i])

        linear_depth_decrementer(q)
        output("out", collect_measurements(measure_array(q)))

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=8).with_seed(29).with_shots(10)
    for value in [0, 1, 13, 30, 31]:
        input_bits = _little_endian_bits(value, 5)
        expected = _output_string((value - 1) % 32, 5)

        emulator_result = emulator.run(bits=input_bits)
        assert emulator_result.collated_counts() == Counter({(("out", expected),): 10})


def test_linear_depth_decrementer_bounded_width() -> None:
    """Bounded decrement only acts on the active low bits."""

    @guppy
    @no_type_check
    def main(bits: array[bool, 5]) -> None:
        q = qarray(5)

        for i in range(5):
            if bits[i]:
                x(q[i])

        linear_depth_decrementer(q, 14)
        output("out", collect_measurements(measure_array(q)))

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=7).with_seed(37).with_shots(10)
    for value in [0, 1, 2, 3, 14]:
        input_bits = _little_endian_bits(value, 5)
        expected = _output_string((value - 1) % 16, 5)

        emulator_result = emulator.run(bits=input_bits)
        assert emulator_result.collated_counts() == Counter({(("out", expected),): 10})


@pytest.mark.symbolic_alternative
def test_cntrl_linear_depth_decrementer() -> None:
    """Decrement only when the control qubit is set."""

    @guppy
    @no_type_check
    def main(bits: array[bool, 5], control_on: bool) -> None:
        control = qubit()
        q = qarray(5)

        if control_on:
            x(control)
        for i in range(5):
            if bits[i]:
                x(q[i])

        cntrl_linear_depth_decrementer(control, q)
        output("ctrl", measure(control).read())
        output("out", collect_measurements(measure_array(q)))

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=10).with_seed(31).with_shots(10)
    for control_on in [False, True]:
        for value in [0, 1, 13, 30, 31]:
            input_bits = _little_endian_bits(value, 5)
            expected_value = (value - 1) % 32 if control_on else value
            expected = _output_string(expected_value, 5)

            emulator_result = emulator.run(bits=input_bits, control_on=control_on)
            expected_ctrl = "1" if control_on else "0"
            assert emulator_result.collated_counts() == Counter(
                {(("ctrl", expected_ctrl), ("out", expected)): 10}
            )


def test_cntrl_linear_depth_decrementer_bounded_width() -> None:
    """Controlled bounded decrement respects the active-width optimization."""

    @guppy
    @no_type_check
    def main(bits: array[bool, 5], control_on: bool) -> None:
        control = qubit()
        q = qarray(5)

        if control_on:
            x(control)
        for i in range(5):
            if bits[i]:
                x(q[i])

        cntrl_linear_depth_decrementer(control, q, 14)
        output("ctrl", measure(control).read())
        output("out", collect_measurements(measure_array(q)))

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=9).with_seed(41).with_shots(10)
    for control_on in [False, True]:
        for value in [0, 1, 2, 3, 14]:
            input_bits = _little_endian_bits(value, 5)
            expected_value = (value - 1) % 16 if control_on else value
            expected = _output_string(expected_value, 5)

            emulator_result = emulator.run(bits=input_bits, control_on=control_on)
            expected_ctrl = "1" if control_on else "0"
            assert emulator_result.collated_counts() == Counter(
                {(("ctrl", expected_ctrl), ("out", expected)): 10}
            )
