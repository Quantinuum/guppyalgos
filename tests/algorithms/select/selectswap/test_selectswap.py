"""Tests for the SelectSWAP data lookup primitive.

Tests cover input validation, basis-state data-entry selection and padding,
and coherent target-register permutations with relative-phase preservation.
"""

from math import ceil, log2
from typing import cast, no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.emulator import EmulatorInstance
from guppylang.std.builtins import array, output
from guppylang.std.debug import state_output
from guppylang.std.quantum import (
    collect_measurements,
    discard_array,
    h,
    measure_array,
    toffoli,
    x,
)
from selene_sim import Quest

from guppyalgos.primitives.measurement import discard_nested_array
from guppyalgos.algorithms.select.selectswap import selectswap
from guppyalgos.utils import int_to_bits, qarray
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    extract_state_branches_in_superposition,
)


# Basis-state emulator helper
def _build_basis_emulator(
    data_inputs: list[list[bool]],
    n_target_registers: int,
) -> tuple[EmulatorInstance, int, int]:
    """Compile one basis-state program for repeated ``(i, j)`` inputs.

    Passing the index bits as runtime arguments lets the basis-state tests reuse
    one compiled emulator for every data-entry selection. The derived ``i`` and
    ``j`` widths are returned with it for preparing and checking those inputs.
    """
    n_rows = ceil(len(data_inputs) / n_target_registers)
    n_index_i_qubits = ceil(log2(n_rows))
    n_index_j_qubits = ceil(log2(n_target_registers))
    width = len(data_inputs[0])
    operation = selectswap(
        data_inputs,
        n_target_registers,
        comp_and_op=toffoli,
        uncomp_and_op=toffoli,
    )

    @guppy
    @no_type_check
    def main(
        index_i_bits: array[bool, n_index_i_qubits],
        index_j_bits: array[bool, n_index_j_qubits],
    ) -> None:
        index_i_qreg = qarray(n_index_i_qubits)
        index_j_qreg = qarray(n_index_j_qubits)
        target_qregs = array(qarray(width) for _ in range(n_target_registers))

        for bit in range(n_index_i_qubits):
            if index_i_bits[bit]:
                x(index_i_qreg[bit])
        for bit in range(n_index_j_qubits):
            if index_j_bits[bit]:
                x(index_j_qreg[bit])

        operation(index_i_qreg, index_j_qreg, target_qregs)

        output("index_i", collect_measurements(measure_array(index_i_qreg)))
        output("index_j", collect_measurements(measure_array(index_j_qreg)))
        target_0 = target_qregs.take(0)
        output("target_0", collect_measurements(measure_array(target_0)))
        for register in range(1, len(target_qregs)):
            discard_array(target_qregs.take(register))
        target_qregs.discard_all_taken()

    # The unary-iteration AND cascade requires n_index_i_qubits - 1 work qubits.
    n_work_qubits = n_index_i_qubits - 1
    n_qubits = (
        n_index_i_qubits + n_index_j_qubits + n_target_registers * width + n_work_qubits
    )
    return main.emulator(n_qubits=n_qubits), n_index_i_qubits, n_index_j_qubits


# Input validation tests
@pytest.mark.parametrize(
    ("data_inputs", "n_target_registers", "message"),
    [
        pytest.param([], 2, "non-empty data_inputs", id="empty-data-inputs"),
        pytest.param(
            [[True], [False], [True], [False], []],
            2,
            "non-empty list of bool",
            id="empty-data-entry",
        ),
        pytest.param(
            [[True], [False], [True, False], [False], [True]],
            2,
            "same width",
            id="unequal-entry-widths",
        ),
        pytest.param(
            [[True], [False], [True], [False], [True]],
            1,
            "at least two",
            id="too-few-target-registers",
        ),
        pytest.param(
            [[True], [False], [True], [False]],
            2,
            "at least three QROM rows",
            id="too-few-qrom-rows",
        ),
    ],
)
def test_selectswap_invalid_configuration(
    data_inputs: list[list[bool]],
    n_target_registers: int,
    message: str,
) -> None:
    """Reject invalid data inputs and unsupported target-register layouts."""
    with pytest.raises(ValueError, match=message):
        selectswap(data_inputs, n_target_registers)


def test_selectswap_non_boolean_data() -> None:
    """Require Boolean values in every data entry rather than truthy integers."""
    data_inputs = cast(list[list[bool]], [[True], [False], [1], [False], [True]])

    with pytest.raises(ValueError, match="list of bool"):
        selectswap(data_inputs, 2)


@pytest.mark.parametrize(
    "n_target_registers",
    [
        pytest.param(cast(int, 2.5), id="float"),
        pytest.param(True, id="boolean"),
    ],
)
def test_selectswap_invalid_target_register_count(
    n_target_registers: int,
) -> None:
    """Reject floating-point and Boolean target-register counts."""
    with pytest.raises(ValueError, match="must be an integer"):
        selectswap([[True], [False], [True], [False], [True]], n_target_registers)


# SelectSWAP behavior tests
def test_selectswap_partial_row_non_power_of_two() -> None:
    """Check partial-row padding and every valid ``j`` branch for ``k = 3``.

    For ``L = 7``, the third QROM row contains one data entry and two all-False
    padding entries. Every valid data-entry index must preserve both index registers
    and move the selected data entry to target register zero. The unused binary
    ``i = 3`` branch must leave target register zero clear for each valid ``j < 3``.
    The invalid ``j = 3`` branch is excluded because SelectSWAP requires callers to
    put no amplitude on ``j >= k``.
    """
    data_inputs = [int_to_bits(i + 1, 3) for i in range(7)]
    n_target_registers = 3
    emulator, n_index_i_qubits, n_index_j_qubits = _build_basis_emulator(
        data_inputs, n_target_registers
    )

    assert (n_index_i_qubits, n_index_j_qubits) == (2, 2)
    # Decompose each data-entry index as data_entry_index = k * i + j.
    for data_entry_index, expected_data in enumerate(data_inputs):
        index_i = data_entry_index // n_target_registers
        index_j = data_entry_index % n_target_registers
        index_i_bits = int_to_bits(index_i, n_index_i_qubits)
        index_j_bits = int_to_bits(index_j, n_index_j_qubits)
        shot = emulator.run(
            index_i_bits=index_i_bits,
            index_j_bits=index_j_bits,
        ).collated_shots()[0]

        assert shot["index_i"][0] == index_i_bits
        assert shot["index_j"][0] == index_j_bits
        assert shot["target_0"][0] == expected_data

    # The index pair (i, j) = (2, 2) selects the final row's all-False entry.
    padded_shot = emulator.run(
        index_i_bits=int_to_bits(2, n_index_i_qubits),
        index_j_bits=int_to_bits(2, n_index_j_qubits),
    ).collated_shots()[0]
    assert padded_shot["target_0"][0] == [False, False, False]

    # Unary iteration selects no QROM row for the unused binary i = 3 branch.
    for index_j in range(n_target_registers):
        invalid_i_shot = emulator.run(
            index_i_bits=int_to_bits(3, n_index_i_qubits),
            index_j_bits=int_to_bits(index_j, n_index_j_qubits),
        ).collated_shots()[0]
        assert invalid_i_shot["target_0"][0] == [False, False, False]


def test_selectswap_coherent_statevector() -> None:
    """Check coherent branches, full target permutations, and relative phases.

    Uniform superpositions over ``i`` and ``j`` produce eight branches. Each
    branch must have probability ``1 / 8``, contain the complete loaded row after
    SwapUp, and have the same phase as every other branch.
    """
    data_inputs = [int_to_bits(i, 3) for i in range(8)]
    n_target_registers = 2
    n_rows = 4
    n_index_i_qubits = 2
    n_index_j_qubits = 1
    width = 3
    operation = selectswap(data_inputs, n_target_registers)

    @guppy
    @no_type_check
    def main() -> None:
        index_i_qreg = qarray(n_index_i_qubits)
        index_j_qreg = qarray(n_index_j_qubits)
        target_qregs = array(qarray(width) for _ in range(n_target_registers))

        for bit in range(n_index_i_qubits):
            h(index_i_qreg[bit])
        for bit in range(n_index_j_qubits):
            h(index_j_qreg[bit])

        operation(index_i_qreg, index_j_qreg, target_qregs)

        state_output(
            "branch",
            index_i_qreg[0],
            index_i_qreg[1],
            index_j_qreg[0],
        )
        state_output(
            "targets",
            target_qregs[0][0],
            target_qregs[0][1],
            target_qregs[0][2],
            target_qregs[1][0],
            target_qregs[1][1],
            target_qregs[1][2],
        )
        discard_array(index_i_qreg)
        discard_array(index_j_qreg)
        discard_nested_array(target_qregs)

    # The unary-iteration AND cascade requires n_index_i_qubits - 1 work qubits.
    n_work_qubits = n_index_i_qubits - 1
    n_qubits = (
        n_index_i_qubits + n_index_j_qubits + n_target_registers * width + n_work_qubits
    )
    run = main.emulator(n_qubits=n_qubits).with_seed(42).run()
    states = Quest.extract_states_dict(run.results[0].entries)
    branch_bitstrings = [
        int_to_bits(index_i, n_index_i_qubits) + int_to_bits(index_j, n_index_j_qubits)
        for index_i in range(n_rows)
        for index_j in range(n_target_registers)
    ]
    branches = extract_state_branches_in_superposition(
        states,
        branch_tag="branch",
        result_tags=["targets"],
        branch_bitstrings=branch_bitstrings,
    )

    expected_probability = 1 / (n_rows * n_target_registers)
    phases: list[complex] = []
    for index_i in range(n_rows):
        row_start = n_target_registers * index_i
        row = data_inputs[row_start : row_start + n_target_registers]
        for index_j in range(n_target_registers):
            branch_bits = int_to_bits(index_i, n_index_i_qubits) + int_to_bits(
                index_j, n_index_j_qubits
            )
            projected = branches[tuple(branch_bits)]
            # For k = 2, SwapUp leaves the row unchanged for j = 0 and swaps the
            # two registers for j = 1.
            expected_registers = row if index_j == 0 else list(reversed(row))
            expected_bits = expected_registers[0] + expected_registers[1]
            expected_state = np.zeros(2 ** (n_target_registers * width))
            # Convert the little-endian target bits to a basis-state index.
            expected_index = sum(
                2**bit_position
                for bit_position, value in enumerate(expected_bits)
                if value
            )
            expected_state[expected_index] = 1.0

            assert np.allclose(projected.probability, expected_probability)
            assert_allclose_ignorephase(projected.state.state, expected_state)
            amplitude = projected.state.state[expected_index]
            # Normalize the selected amplitude to retain only its phase.
            phases.append(amplitude / abs(amplitude))

    # Global phase is irrelevant, but relative phase between branches is not.
    np.testing.assert_allclose(phases, np.full(len(phases), phases[0]))


def test_selectswap_uncompute_returns_targets_to_zero() -> None:
    """Check swapup followed by uncompute clears every target register."""
    data_inputs = [int_to_bits(i, 3) for i in range(8)]
    n_target_registers = 2
    n_index_i_qubits = 2
    n_index_j_qubits = 1
    width = 3
    operation = selectswap(data_inputs, n_target_registers)
    uncompute = selectswap(data_inputs, n_target_registers, uncompute=True)

    @guppy
    @no_type_check
    def main() -> None:
        index_i_qreg = qarray(n_index_i_qubits)
        index_j_qreg = qarray(n_index_j_qubits)
        target_qregs = array(qarray(width) for _ in range(n_target_registers))

        for bit in range(n_index_i_qubits):
            h(index_i_qreg[bit])
        for bit in range(n_index_j_qubits):
            h(index_j_qreg[bit])

        operation(index_i_qreg, index_j_qreg, target_qregs)
        uncompute(index_i_qreg, index_j_qreg, target_qregs)

        state_output(
            "targets",
            target_qregs[0][0],
            target_qregs[0][1],
            target_qregs[0][2],
            target_qregs[1][0],
            target_qregs[1][1],
            target_qregs[1][2],
        )
        discard_array(index_i_qreg)
        discard_array(index_j_qreg)
        discard_nested_array(target_qregs)

    n_work_qubits = n_index_i_qubits - 1
    n_qubits = (
        n_index_i_qubits + n_index_j_qubits + n_target_registers * width + n_work_qubits
    )
    run = main.emulator(n_qubits=n_qubits).with_seed(42).run()
    states = Quest.extract_states_dict(run.results[0].entries)

    expected_zero = np.zeros(2 ** (n_target_registers * width))
    expected_zero[0] = 1.0
    assert_allclose_ignorephase(
        states["targets"].get_single_state(),
        expected_zero,
    )
