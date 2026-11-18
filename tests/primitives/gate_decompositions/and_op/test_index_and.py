"""Tests for indexed AND unitaries and their modifiers."""

import pytest

from guppylang.defs import GuppyFunctionDefinition
from guppylang.decorator import guppy
from guppylang.std.builtins import control, dagger, output
from guppylang.std.quantum import (
    collect_measurements,
    measure,
    measure_array,
    qubit,
    toffoli,
    x,
)

from guppyalgos.primitives.gate_decompositions.and_op import (
    index_and,
    temp_and,
)
from guppyalgos.utils import int_to_bits, qarray

from typing import no_type_check


@pytest.mark.parametrize("and_operation", [toffoli, temp_and])
@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_indexed_and(
    i: int,
    and_operation: GuppyFunctionDefinition,
) -> None:
    """Compute an AND when both controls match the requested index."""
    binary_index = int_to_bits(i, 2)

    @guppy
    @no_type_check
    def main() -> None:
        index_qreg = qarray(2)
        target = qubit()

        for bit in range(2):
            if binary_index[bit]:
                x(index_qreg[bit])

        index_and(
            target,
            index_qreg[0],
            binary_index[0],
            index_qreg[1],
            binary_index[1],
            and_operation,
        )

        output("index", collect_measurements(measure_array(index_qreg)))
        output("target", measure(target).read())

    result = main.emulator(n_qubits=3).with_seed(451).run()
    shot = result.collated_shots()[0]
    assert shot["index"][0] == binary_index
    assert shot["target"][0] == 1


@pytest.mark.parametrize("and_operation", [toffoli, temp_and])
def test_indexed_and_daggered(
    and_operation: GuppyFunctionDefinition,
) -> None:
    """Uncompute a matching indexed AND with its custom dagger."""

    @guppy
    @no_type_check
    def main() -> None:
        index_qreg = qarray(2)
        x(index_qreg[1])
        target = qubit()
        x(target)

        with dagger:
            index_and(target, index_qreg[0], False, index_qreg[1], True, and_operation)

        output("index", collect_measurements(measure_array(index_qreg)))
        output("target", measure(target).read())

    result = main.emulator(n_qubits=3).with_seed(451).run()
    shot = result.collated_shots()[0]
    assert shot["index"][0] == [False, True]
    assert shot["target"][0] == 0


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("and_operation", [toffoli, temp_and])
def test_indexed_and_controlled(
    enabled: bool,
    and_operation: GuppyFunctionDefinition,
) -> None:
    """Apply an indexed AND only when its additional control is enabled."""

    @guppy
    @no_type_check
    def main() -> None:
        index_qreg = qarray(2)
        x(index_qreg[1])
        controls = qarray(1)
        if enabled:
            x(controls[0])
        target = qubit()

        with control(controls):
            index_and(target, index_qreg[0], False, index_qreg[1], True, and_operation)

        output("index", collect_measurements(measure_array(index_qreg)))
        output("controls", collect_measurements(measure_array(controls)))
        output("target", measure(target).read())

    result = main.emulator(n_qubits=5).with_seed(451).run()
    shot = result.collated_shots()[0]
    assert shot["target"][0] == int(enabled)


@pytest.mark.parametrize("and_operation", [toffoli, temp_and])
def test_indexed_and_controlled_daggered(
    and_operation: GuppyFunctionDefinition,
) -> None:
    """Uncompute a matching indexed AND with its controlled custom dagger."""

    @guppy
    @no_type_check
    def main() -> None:
        index_qreg = qarray(2)
        x(index_qreg[1])
        controls = qarray(1)
        x(controls[0])
        target = qubit()
        x(target)

        with control(controls), dagger:
            index_and(target, index_qreg[0], False, index_qreg[1], True, and_operation)

        output("index", collect_measurements(measure_array(index_qreg)))
        output("controls", collect_measurements(measure_array(controls)))
        output("target", measure(target).read())

    result = main.emulator(n_qubits=5).with_seed(451).run()
    shot = result.collated_shots()[0]
    assert shot["index"][0] == [False, True]
    assert shot["controls"][0] == [True]
    assert shot["target"][0] == 0
