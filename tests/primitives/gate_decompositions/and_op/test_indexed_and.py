"""Tests for indexed AND unitaries and their modifiers."""

import numpy as np
import pytest

from guppylang.defs import GuppyFunctionDefinition
from guppylang.decorator import guppy
from guppylang.std.builtins import array, control, dagger, output
from guppylang.std.quantum import (
    collect_measurements,
    cx,
    h,
    measure,
    measure_array,
    qubit,
    toffoli,
    x,
)

from guppyalgos.primitives.gate_decompositions.and_op import (
    indexed_and,
    temp_and,
)
from guppyalgos.testing import assert_unitary_modifiers
from guppyalgos.utils import int_to_bits, qarray

from typing import no_type_check


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_indexed_toffoli_modifiers(index: int) -> None:
    """Compare every modifier against the indexed Toffoli matrix."""
    bits = int_to_bits(index, 2)
    first_bit, second_bit = bits

    @guppy(unitary=True)
    @no_type_check
    def operation(qs: array[qubit, 3]) -> None:
        indexed_and(qs[2], qs[0], first_bit, qs[1], second_bit, toffoli)

    expected = np.eye(8)
    matching = 4 * int(bits[0]) + 2 * int(bits[1])
    expected[matching, matching] = 0
    expected[matching + 1, matching + 1] = 0
    expected[matching, matching + 1] = 1
    expected[matching + 1, matching] = 1
    assert_unitary_modifiers(operation, 3, expected_unitary=expected)


@pytest.mark.parametrize(("index", "n_controls"), [(0, 1), (1, 2), (2, 2), (3, 1)])
@pytest.mark.parametrize("inverse", [False, True])
def test_indexed_and_coherent(index: int, n_controls: int, inverse: bool) -> None:
    """Check the promised-input isometry and measurement cleanup coherently."""
    first_bit, second_bit = int_to_bits(index, 2)
    n_references = n_controls + 2

    @guppy(unitary=True)
    @no_type_check
    def operation(
        inputs: array[qubit, 2], target: qubit, controls: array[qubit, n_controls]
    ) -> None:
        with control(controls):
            indexed_and(
                target, inputs[0], first_bit, inputs[1], second_bit, temp_and
            )

    @guppy
    @no_type_check
    def reference(
        inputs: array[qubit, 2],
        target: qubit,
        controls: array[qubit, n_controls],
        work: array[qubit, n_controls],
    ) -> None:
        if not first_bit:
            x(inputs[0])
        if not second_bit:
            x(inputs[1])
        toffoli(controls[0], inputs[0], work[0])
        for i in range(1, n_controls):
            toffoli(controls[i], work[i - 1], work[i])
        toffoli(work[n_controls - 1], inputs[1], target)
        for i in range(n_controls - 1, 0, -1):
            toffoli(controls[i], work[i - 1], work[i])
        toffoli(controls[0], inputs[0], work[0])
        if not first_bit:
            x(inputs[0])
        if not second_bit:
            x(inputs[1])

    @guppy
    @no_type_check
    def main() -> None:
        controls = qarray(n_controls)
        inputs = qarray(2)
        references = qarray(n_references)
        work = qarray(n_controls)
        target = qubit()

        for i in range(n_controls):
            h(controls[i])
            cx(controls[i], references[i])
        for i in range(2):
            h(inputs[i])
            cx(inputs[i], references[n_controls + i])

        if inverse:
            reference(inputs, target, controls, work)
            with dagger:
                operation(inputs, target, controls)
        else:
            operation(inputs, target, controls)
            reference(inputs, target, controls, work)

        for i in range(n_controls):
            cx(controls[i], references[i])
            h(controls[i])
        for i in range(2):
            cx(inputs[i], references[n_controls + i])
            h(inputs[i])

        output("controls", collect_measurements(measure_array(controls)))
        output("inputs", collect_measurements(measure_array(inputs)))
        output("references", collect_measurements(measure_array(references)))
        output("work", collect_measurements(measure_array(work)))
        output("target", measure(target).read())

    shots = main.emulator(4 * n_controls + 5).with_shots(32).run()
    for shot in shots.collated_shots():
        assert shot["controls"][0] == [False] * n_controls
        assert shot["inputs"][0] == [False] * 2
        assert shot["references"][0] == [False] * (n_controls + 2)
        assert shot["work"][0] == [False] * n_controls
        assert shot["target"][0] == 0


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

        indexed_and(
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
            indexed_and(
                target, index_qreg[0], False, index_qreg[1], True, and_operation
            )

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
            indexed_and(
                target, index_qreg[0], False, index_qreg[1], True, and_operation
            )

        output("index", collect_measurements(measure_array(index_qreg)))
        output("controls", collect_measurements(measure_array(controls)))
        output("target", measure(target).read())

    result = main.emulator(n_qubits=5).with_seed(451).run()
    shot = result.collated_shots()[0]
    assert shot["index"][0] == [False, True]
    assert shot["controls"][0] == [enabled]
    assert shot["target"][0] == int(enabled)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("and_operation", [toffoli, temp_and])
def test_indexed_and_controlled_daggered(
    enabled: bool,
    and_operation: GuppyFunctionDefinition,
) -> None:
    """Check both branches of the controlled custom dagger."""
    target_initial = enabled or and_operation is toffoli

    @guppy
    @no_type_check
    def main() -> None:
        index_qreg = qarray(2)
        x(index_qreg[1])
        controls = qarray(1)
        if enabled:
            x(controls[0])
        target = qubit()
        if target_initial:
            x(target)

        with control(controls), dagger:
            indexed_and(
                target, index_qreg[0], False, index_qreg[1], True, and_operation
            )

        output("index", collect_measurements(measure_array(index_qreg)))
        output("controls", collect_measurements(measure_array(controls)))
        output("target", measure(target).read())

    result = main.emulator(n_qubits=5).with_seed(451).run()
    shot = result.collated_shots()[0]
    assert shot["index"][0] == [False, True]
    assert shot["controls"][0] == [enabled]
    assert shot["target"][0] == int(target_initial and not enabled)
