"""Tests for the conditionally-clean-ancilla incrementer."""

from collections import Counter
from typing import no_type_check

import pytest
from guppylang import guppy
from guppylang.std.builtins import array, control, dagger, output
from guppylang.std.quantum import (
    collect_measurements,
    measure,
    measure_array,
    x,
    discard_array,
    qubit,
)
from guppyalgos.utils import qarray

from guppyalgos.primitives.arithmetic.incrementer.incrementer_cca import (
    cat_cca,
    cca_incrementer,
    cca_decrementer,
    cntrl_cca_incrementer,
    cntrl_cca_decrementer,
    required_clean_ancillas,
)


def _little_endian_bits(value: int, num_bits: int) -> list[bool]:
    return [bool((value >> i) & 1) for i in range(num_bits)]


def _output_string(value: int, num_bits: int) -> str:
    bits = _little_endian_bits(value, num_bits)
    return "".join("1" if bit else "0" for bit in bits)


@pytest.mark.parametrize("num_bits", [4, 5, 7, 12, 19])
def test_cca_incrementer_compiles(num_bits: int) -> None:
    """Test CCA incremeter compiles without ownership or borrowing errors."""

    @guppy
    @no_type_check
    def main() -> None:
        q = qarray(num_bits)

        cca_incrementer(q)

        discard_array(q)

    main.compile()


@pytest.mark.parametrize("num_bits", [5, 7, 10, 20, 50, 100, 200, 500])
def test_incrementer_does_not_panic(num_bits: int) -> None:
    """Test CCA incrementer runs without runtime panics."""
    num_ancillas = required_clean_ancillas(num_bits)

    @guppy
    @no_type_check
    def main() -> None:
        q = qarray(num_bits)

        cca_incrementer(q)

        discard_array(q)

    (main.emulator(n_qubits=num_bits + num_ancillas).coinflip_sim().run())


@pytest.mark.parametrize(
    ("num_bits", "values"),
    [
        (5, [0, 1, 13, 30, 31]),
        (7, [0, 1, 63, 126, 127]),
    ],
)
def test_cca_incrementer(num_bits: int, values: list[int]) -> None:
    """Increment an m-bit register using conditionally clean ancillas."""
    num_ancillas = required_clean_ancillas(num_bits)

    @guppy
    @no_type_check
    def main(bits: array[bool, num_bits]) -> None:
        q = qarray(num_bits)

        for i in range(num_bits):
            if bits[i]:
                x(q[i])

        cca_incrementer(q)

        output(
            "out",
            collect_measurements(measure_array(q)),
        )

    # Group values within a test so compilation is reused even with xdist.
    emulator = (
        main.emulator(n_qubits=num_bits + num_ancillas).with_seed(43).with_shots(10)
    )
    for value in values:
        expected = _output_string((value + 1) % (1 << num_bits), num_bits)
        result = emulator.run(bits=_little_endian_bits(value, num_bits))
        assert result.collated_counts() == Counter({(("out", expected),): 10}), value


@pytest.mark.parametrize(
    ("num_bits", "values"),
    [
        (5, [0, 1, 13, 30, 31]),
        (7, [0, 1, 63, 126, 127]),
    ],
)
def test_cca_decrementer(num_bits: int, values: list[int]) -> None:
    """Decrement an m-bit register using conditionally clean ancillas."""
    num_ancillas = required_clean_ancillas(num_bits)

    @guppy
    @no_type_check
    def main(bits: array[bool, num_bits]) -> None:
        q = qarray(num_bits)

        for i in range(num_bits):
            if bits[i]:
                x(q[i])

        cca_decrementer(q)

        output(
            "out",
            collect_measurements(measure_array(q)),
        )

    emulator = (
        main.emulator(n_qubits=num_bits + num_ancillas).with_seed(43).with_shots(10)
    )
    for value in values:
        expected = _output_string((value - 1) % (1 << num_bits), num_bits)
        result = emulator.run(bits=_little_endian_bits(value, num_bits))
        assert result.collated_counts() == Counter({(("out", expected),): 10}), value


@pytest.mark.parametrize(
    ("num_bits", "values"),
    [
        (5, [0, 1, 13, 30, 31]),
        (7, [0]),
        (6, [1, 63, 126, 127]),
    ],
)
@pytest.mark.parametrize("dagger", [False, True])
def test_cntrl_cca_incrementer(
    num_bits: int,
    values: list[int],
    dagger: bool,
) -> None:
    """Increment/decrement iff all control qubits are set."""
    num_controls = 1

    num_ancillas_big = required_clean_ancillas(num_bits + num_controls)
    num_ancillas_small = required_clean_ancillas(num_controls)
    num_ancillas = num_ancillas_big + num_ancillas_small + 1

    @guppy
    @no_type_check
    def main(bits: array[bool, num_bits], control_on: bool) -> None:
        control = qubit()
        q = qarray(num_bits)

        if control_on:
            x(control)

        for i in range(num_bits):
            if bits[i]:
                x(q[i])

        if not dagger:
            cntrl_cca_incrementer(control, q)
        else:
            cntrl_cca_decrementer(control, q)

        output("ctrl", measure(control).read())
        output("out", collect_measurements(measure_array(q)))

    emulator = (
        main.emulator(n_qubits=(num_bits + num_controls + num_ancillas))
        .with_seed(41)
        .with_shots(10)
    )
    for control_on in [False, True]:
        for value in values:
            value %= 1 << num_bits
            delta = -1 if dagger else 1
            expected_value = (value + delta) % (1 << num_bits) if control_on else value
            expected = _output_string(expected_value, num_bits)
            expected_ctrl = _output_string(control_on, num_controls)
            result = emulator.run(
                bits=_little_endian_bits(value, num_bits), control_on=control_on
            )
            assert result.collated_counts() == Counter(
                {(("ctrl", expected_ctrl), ("out", expected)): 10}
            ), (value, control_on, dagger)


def test_cat_cca_modifier_compilation() -> None:
    """Compile added controls and adjoints for the incrementer CCA catalyst."""

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, 3], cs: array[qubit, 2]) -> None:
        with control(cs), dagger:
            cat_cca(qs[0], qs[1], qs[2])

    circuit.compile_function()
