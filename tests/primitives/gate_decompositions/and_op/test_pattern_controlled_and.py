"""Tests for indexed AND unitaries and their modifiers."""

import numpy as np
import pytest

from guppylang.defs import GuppyFunctionDefinition
from guppylang.decorator import guppy
from guppylang.std.builtins import array, control, dagger, output
from guppylang.std.debug import state_output
from guppylang.std.quantum import (
    collect_measurements,
    cx,
    discard,
    discard_array,
    h,
    measure,
    measure_array,
    qubit,
    toffoli,
    x,
)
from selene_sim import Quest

from guppyalgos.primitives.gate_decompositions.and_op import (
    pattern_controlled_and,
    temp_and,
)
from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx
from guppyalgos.testing import (
    assert_unitary_modifiers,
    get_total_state_on_only_specified_registers,
)
from guppyalgos.utils import int_to_bits, qarray

from typing import no_type_check


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_indexed_toffoli_modifiers(index: int) -> None:
    """Compare the Toffoli-backed wrapper with an independent indexed matrix.

    For each index, check all input states in the forward, dagger, controlled,
    and controlled-dagger modes, including inactive controls and relative phases.
    """
    bits = int_to_bits(index, 2)
    first_bit, second_bit = bits

    @guppy(unitary=True)
    @no_type_check
    def operation(qs: array[qubit, 3]) -> None:
        pattern_controlled_and(qs[2], qs[0], first_bit, qs[1], second_bit, toffoli)

    expected = np.eye(8)
    matching = 4 * int(bits[0]) + 2 * int(bits[1])
    expected[matching, matching] = 0
    expected[matching + 1, matching + 1] = 0
    expected[matching, matching + 1] = 1
    expected[matching + 1, matching] = 1
    assert_unitary_modifiers(operation, 3, expected_unitary=expected)


@pytest.mark.parametrize(("index", "n_controls"), [(0, 1), (1, 2), (2, 2), (3, 1)])
@pytest.mark.parametrize("inverse", [False, True])
def test_pattern_controlled_and_coherent(
    index: int, n_controls: int, inverse: bool
) -> None:
    """Compare the controlled temporary AND with an independent CNX circuit.

    Entangled inputs and controls check the promised-input action and its dagger
    for all index patterns, including relative phases and control coherence.
    Multiple runs exercise measurement-based cleanup. Statevectors verify that
    the inputs, controls, references, and target return to zero.
    """
    first_bit, second_bit = int_to_bits(index, 2)
    n_references = n_controls + 2

    @guppy(unitary=True)
    @no_type_check
    def operation(
        inputs: array[qubit, 2], target: qubit, controls: array[qubit, n_controls]
    ) -> None:
        with control(controls):
            pattern_controlled_and(
                target, inputs[0], first_bit, inputs[1], second_bit, temp_and
            )

    @guppy
    @no_type_check
    def reference(
        all_controls: array[qubit, n_references],
        target: qubit,
    ) -> None:
        if not first_bit:
            x(all_controls[n_controls])
        if not second_bit:
            x(all_controls[n_controls + 1])
        cnx(all_controls, target)
        if not first_bit:
            x(all_controls[n_controls])
        if not second_bit:
            x(all_controls[n_controls + 1])

    @guppy
    @no_type_check
    def main() -> None:
        all_controls = qarray(n_references)
        references = qarray(n_references)
        target = qubit()

        for i in range(n_references):
            h(all_controls[i])
            cx(all_controls[i], references[i])

        if inverse:
            reference(all_controls, target)

        controls = array(all_controls.take(i) for i in range(n_controls))
        inputs = array(all_controls.take(n_controls + i) for i in range(2))
        if inverse:
            with dagger:
                operation(inputs, target, controls)
        else:
            operation(inputs, target, controls)
        i = 0
        for control_qubit in controls:
            all_controls.put(control_qubit, i)
            i += 1
        for input_qubit in inputs:
            all_controls.put(input_qubit, i)
            i += 1

        if not inverse:
            reference(all_controls, target)

        for i in range(n_references):
            cx(all_controls[i], references[i])
            h(all_controls[i])

        state_output("controls", all_controls)
        state_output("references", references)
        state_output("target", target)
        discard_array(all_controls)
        discard_array(references)
        discard(target)

    shots = main.emulator(4 * n_controls + 5).with_shots(32).run()
    expected = np.zeros(2 ** (2 * n_references + 1))
    expected[0] = 1
    for shot in shots.results:
        states = Quest.extract_states_dict(shot.entries)
        state, _ = get_total_state_on_only_specified_registers(
            states, ["controls", "references", "target"]
        )
        np.testing.assert_allclose(np.abs(state.state), expected, atol=1e-8)


@pytest.mark.parametrize("and_operation", [toffoli, temp_and])
@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_pattern_controlled_and(
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

        pattern_controlled_and(
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
def test_pattern_controlled_and_daggered(
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
            pattern_controlled_and(
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
def test_pattern_controlled_and_controlled(
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
            pattern_controlled_and(
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
def test_pattern_controlled_and_controlled_daggered(
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
            pattern_controlled_and(
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
