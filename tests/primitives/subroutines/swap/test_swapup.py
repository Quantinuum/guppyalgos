"""Tests for qubit and register SwapUp operations.

Basis-state tests compare classical bit permutations. Coherent tests project
index branches and compare them with references made from uncontrolled
CX-based swaps. The suite also tests the public ``swapup_linear`` overload.
"""

from math import ceil, log2
from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, comptime, result
from guppylang.std.debug import state_output
from guppylang.std.quantum import (
    collect_measurements,
    cx,
    discard_array,
    h,
    measure_array,
    qubit,
    ry,
    x,
)
from selene_sim import Quest

from guppyalgos.primitives.measurement import discard_nested_array
from guppyalgos.primitives.subroutines.swap import swapup_linear
from guppyalgos.primitives.subroutines.swap.swapup import (
    swapup_linear_qubit,
    swapup_linear_register,
)
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_total_state_on_only_specified_registers,
    project_state_onto_bitstring,
    switch_endianness,
)
from guppyalgos.utils import int_to_bits


# Expected-state helpers
def _swapup_linear_pairs_for_index(
    index_value: int, n_state_qubits: int
) -> list[tuple[int, int]]:
    """Build test-side swap pairs for a classical little-endian SwapUp index.

    Layers run from the largest stride to the smallest. For a non-power-of-two
    state register, the top layer has ``n_state_qubits - stride`` pairs.
    """
    n_index_qubits = ceil(log2(n_state_qubits))
    swap_pairs = []

    # This is deliberately separate from _swapup_linear_schedule for cross-checking.
    for bit_pos in range(n_index_qubits - 1, -1, -1):
        stride = 2**bit_pos
        if bit_pos == n_index_qubits - 1:
            n_swaps = n_state_qubits - stride
        else:
            n_swaps = stride

        if (index_value >> bit_pos) & 1:
            swap_pairs.extend((lower, lower + stride) for lower in range(n_swaps))

    return swap_pairs


def _expected_swapup_linear_state(
    index_value: int, state_bits: list[bool]
) -> list[bool]:
    """Apply direct SwapUp schedule to classical basis bits."""
    expected = list(state_bits)

    for lower, upper in _swapup_linear_pairs_for_index(index_value, len(state_bits)):
        expected[lower], expected[upper] = expected[upper], expected[lower]

    return expected


def _swapup_linear_rotation_angles(n_state_qubits: int) -> list[float]:
    """Use distinct non-basis angles so qubit permutations are observable."""
    return [0.17 + 0.11 * i for i in range(n_state_qubits)]


def _swapup_linear_register_rotation_angles(
    n_state_registers: int, width: int
) -> list[list[float]]:
    """Use distinct angles for every qubit in every state register."""
    return [
        [0.13 + 0.17 * register + 0.07 * bit for bit in range(width)]
        for register in range(n_state_registers)
    ]


def _expected_swapup_linear_branch_state(
    index_value: int, n_state_qubits: int
) -> np.ndarray:
    """Build an independent expected branch using unconditional swaps."""
    rotation_angles = _swapup_linear_rotation_angles(n_state_qubits)
    swap_pairs = _swapup_linear_pairs_for_index(index_value, n_state_qubits)

    # Guppy cannot infer the tuple type in comptime([]), so omit an empty loop.
    if swap_pairs:

        @guppy
        @no_type_check
        def main() -> None:
            state_qreg = array(qubit() for _ in range(comptime(n_state_qubits)))

            for bit, theta in comptime(list(enumerate(rotation_angles))):
                ry(state_qreg[bit], angle(theta))

            for lower, upper in comptime(swap_pairs):
                cx(state_qreg[lower], state_qreg[upper])
                cx(state_qreg[upper], state_qreg[lower])
                cx(state_qreg[lower], state_qreg[upper])

            state_output("state", state_qreg)
            discard_array(state_qreg)

    else:

        @guppy
        @no_type_check
        def main() -> None:
            state_qreg = array(qubit() for _ in range(comptime(n_state_qubits)))

            for bit, theta in comptime(list(enumerate(rotation_angles))):
                ry(state_qreg[bit], angle(theta))

            state_output("state", state_qreg)
            discard_array(state_qreg)

    result_state = Quest.extract_states_dict(
        main.emulator(n_qubits=n_state_qubits).with_seed(42).run().results[0].entries
    )["state"].get_single_state()
    return switch_endianness(result_state)


def _expected_register_branch_state(index_value: int) -> np.ndarray:
    """Build an independent reference state for one register-SwapUp branch."""
    n_state_registers = 4
    width = 2
    rotation_angles = _swapup_linear_register_rotation_angles(n_state_registers, width)
    swap_pairs = _swapup_linear_pairs_for_index(index_value, n_state_registers)

    # Guppy cannot infer the tuple type in comptime([]), so omit an empty loop.
    if swap_pairs:

        @guppy
        @no_type_check
        def main() -> None:
            state_qregs = array(
                array(qubit() for _ in range(comptime(width)))
                for _ in range(comptime(n_state_registers))
            )

            for register, bit, theta in comptime(
                [
                    (register, bit, theta)
                    for register, register_angles in enumerate(rotation_angles)
                    for bit, theta in enumerate(register_angles)
                ]
            ):
                ry(state_qregs[register][bit], angle(theta))

            for lower, upper in comptime(swap_pairs):
                for bit in range(comptime(width)):
                    cx(state_qregs[lower][bit], state_qregs[upper][bit])
                    cx(state_qregs[upper][bit], state_qregs[lower][bit])
                    cx(state_qregs[lower][bit], state_qregs[upper][bit])

            state_output(
                "state",
                state_qregs[0][0],
                state_qregs[0][1],
                state_qregs[1][0],
                state_qregs[1][1],
                state_qregs[2][0],
                state_qregs[2][1],
                state_qregs[3][0],
                state_qregs[3][1],
            )
            discard_nested_array(state_qregs)

    else:

        @guppy
        @no_type_check
        def main() -> None:
            state_qregs = array(
                array(qubit() for _ in range(comptime(width)))
                for _ in range(comptime(n_state_registers))
            )

            for register, bit, theta in comptime(
                [
                    (register, bit, theta)
                    for register, register_angles in enumerate(rotation_angles)
                    for bit, theta in enumerate(register_angles)
                ]
            ):
                ry(state_qregs[register][bit], angle(theta))

            state_output(
                "state",
                state_qregs[0][0],
                state_qregs[0][1],
                state_qregs[1][0],
                state_qregs[1][1],
                state_qregs[2][0],
                state_qregs[2][1],
                state_qregs[3][0],
                state_qregs[3][1],
            )
            discard_nested_array(state_qregs)

    result_state = Quest.extract_states_dict(
        main.emulator(n_qubits=n_state_registers * width)
        .with_seed(42)
        .run()
        .results[0]
        .entries
    )["state"].get_single_state()
    return switch_endianness(result_state)


# Guppy test circuits
def _run_swapup_linear_qubit(index_value: int, state_bits: list[bool]) -> dict:
    """Run qubit SwapUp on basis inputs and measure the index and state qubits."""
    n_state_qubits = len(state_bits)
    n_index_qubits = ceil(log2(n_state_qubits))
    index_bits = int_to_bits(index_value, n_index_qubits)

    @guppy
    @no_type_check
    def main() -> None:
        idx = comptime(index_bits)
        state_input = comptime(state_bits)
        index_qreg = array(qubit() for _ in range(comptime(n_index_qubits)))
        state_qreg = array(qubit() for _ in range(comptime(n_state_qubits)))

        for bit in range(comptime(n_index_qubits)):
            if idx[bit]:
                x(index_qreg[bit])

        for bit in range(comptime(n_state_qubits)):
            if state_input[bit]:
                x(state_qreg[bit])

        swapup_linear_qubit(index_qreg, state_qreg)

        result("index", collect_measurements(measure_array(index_qreg)))
        result("state", collect_measurements(measure_array(state_qreg)))

    shots = (
        main.emulator(n_qubits=n_index_qubits + n_state_qubits)
        .with_seed(42)
        .with_shots(1)
        .run()
    )
    return shots.collated_shots()[0]


def _run_swapup_linear_qubit_statevector(n_state_qubits: int):
    """Run qubit SwapUp with a coherent index and capture the full state."""
    n_index_qubits = ceil(log2(n_state_qubits))
    rotation_angles = _swapup_linear_rotation_angles(n_state_qubits)

    @guppy
    @no_type_check
    def main() -> None:
        index_qreg = array(qubit() for _ in range(comptime(n_index_qubits)))
        state_qreg = array(qubit() for _ in range(comptime(n_state_qubits)))

        for bit in range(comptime(n_index_qubits)):
            h(index_qreg[bit])

        for bit, theta in comptime(list(enumerate(rotation_angles))):
            ry(state_qreg[bit], angle(theta))

        swapup_linear_qubit(index_qreg, state_qreg)

        state_output("index", index_qreg)
        state_output("state", state_qreg)
        discard_array(index_qreg)
        discard_array(state_qreg)

    return main.emulator(n_qubits=n_index_qubits + n_state_qubits).with_seed(42).run()


def _run_swapup_linear_register_output_zero(
    index_value: int, state_register_bits: list[list[bool]]
) -> dict:
    """Run register SwapUp and measure the index register and ``state_qregs[0]``."""
    n_state_registers = len(state_register_bits)
    width = len(state_register_bits[0])
    n_index_qubits = ceil(log2(n_state_registers))
    index_bits = int_to_bits(index_value, n_index_qubits)

    @guppy
    @no_type_check
    def main() -> None:
        idx = comptime(index_bits)
        state_input = comptime(state_register_bits)
        index_qreg = array(qubit() for _ in range(comptime(n_index_qubits)))
        state_qregs = array(
            array(qubit() for _ in range(comptime(width)))
            for _ in range(comptime(n_state_registers))
        )

        for bit in range(comptime(n_index_qubits)):
            if idx[bit]:
                x(index_qreg[bit])

        for register in range(comptime(n_state_registers)):
            for bit in range(comptime(width)):
                if state_input[register][bit]:
                    x(state_qregs[register][bit])

        swapup_linear_register(index_qreg, state_qregs)

        result("index", collect_measurements(measure_array(index_qreg)))
        state_0 = state_qregs.take(0)
        result("state_0", collect_measurements(measure_array(state_0)))
        for register in range(1, len(state_qregs)):
            discard_array(state_qregs.take(register))
        state_qregs.discard_all_taken()

    shots = (
        main.emulator(n_qubits=n_index_qubits + n_state_registers * width)
        .with_seed(42)
        .with_shots(1)
        .run()
    )
    return shots.collated_shots()[0]


def _run_swapup_linear_register_statevector():
    """Run register SwapUp with a coherent index and capture all state registers."""
    n_state_registers = 4
    width = 2
    n_index_qubits = ceil(log2(n_state_registers))
    rotation_angles = _swapup_linear_register_rotation_angles(n_state_registers, width)

    @guppy
    @no_type_check
    def main() -> None:
        index_qreg = array(qubit() for _ in range(comptime(n_index_qubits)))
        state_qregs = array(
            array(qubit() for _ in range(comptime(width)))
            for _ in range(comptime(n_state_registers))
        )

        for bit in range(comptime(n_index_qubits)):
            h(index_qreg[bit])

        for register, bit, theta in comptime(
            [
                (register, bit, theta)
                for register, register_angles in enumerate(rotation_angles)
                for bit, theta in enumerate(register_angles)
            ]
        ):
            ry(state_qregs[register][bit], angle(theta))

        swapup_linear_register(index_qreg, state_qregs)

        state_output("index", index_qreg)
        # state_output accepts individual qubits or a one-dimensional qubit array.
        state_output(
            "state",
            state_qregs[0][0],
            state_qregs[0][1],
            state_qregs[1][0],
            state_qregs[1][1],
            state_qregs[2][0],
            state_qregs[2][1],
            state_qregs[3][0],
            state_qregs[3][1],
        )
        discard_array(index_qreg)
        discard_nested_array(state_qregs)

    return (
        main.emulator(n_qubits=n_index_qubits + n_state_registers * width)
        .with_seed(42)
        .run()
    )


# SwapUp behavior tests
def test_swapup_linear_qubit_matches_classical_schedule_non_power_of_two() -> None:
    """Check qubit SwapUp on basis states for each valid index of six state qubits.

    The index is unchanged and the state qubits follow the expected permutation.
    This covers the valid branches of a non-power-of-two state register.
    """
    state_bits = [True, False, True, False, False, True]
    n_index_qubits = ceil(log2(len(state_bits)))

    for index_value in range(len(state_bits)):
        shot = _run_swapup_linear_qubit(index_value, state_bits)

        assert shot["index"][0] == int_to_bits(index_value, n_index_qubits)
        assert shot["state"][0] == _expected_swapup_linear_state(
            index_value, state_bits
        )


def test_swapup_linear_qubit_statevector_branches_match_uncntrl_swaps() -> None:
    """Check all eight coherent qubit SwapUp branches against uncontrolled swaps.

    A uniform index superposition gives each branch probability ``1 / 8``. The
    state must match the independently swapped reference up to global phase.
    """
    n_state_qubits = 8
    n_index_qubits = ceil(log2(n_state_qubits))
    res = _run_swapup_linear_qubit_statevector(n_state_qubits)

    states = Quest.extract_states_dict(res.results[0].entries)
    non_work_state, spec_qubits_dict = get_total_state_on_only_specified_registers(
        states, ["index", "state"]
    )
    index_qubits = spec_qubits_dict["index"]

    for index_value in range(n_state_qubits):
        non_work_state.specified_qubits = index_qubits
        # Projection expects MSB-first bits; SwapUp indices are little-endian.
        projected = project_state_onto_bitstring(
            non_work_state,
            int_to_bits(index_value, n_index_qubits),
        )

        assert np.allclose(projected.probability, 1 / n_state_qubits)

        expected_state = _expected_swapup_linear_branch_state(
            index_value, n_state_qubits
        )
        assert_allclose_ignorephase(projected.state.state, expected_state)


@pytest.mark.parametrize("index_value", [3, 4])
def test_swapup_linear_register_moves_width_three_selected_register_to_zero(
    index_value: int,
) -> None:
    """Check register SwapUp for selected width-three state registers.

    Index ``3`` exercises the middle and low layers; index ``4`` exercises the
    shortened top layer. The index is unchanged and the selected state register
    moves to ``state_qregs[0]``.
    """
    state_register_bits = [
        [False, False, False],
        [True, False, False],
        [False, True, False],
        [True, True, False],
        [False, False, True],
    ]
    n_index_qubits = ceil(log2(len(state_register_bits)))

    shot = _run_swapup_linear_register_output_zero(index_value, state_register_bits)

    assert shot["index"][0] == int_to_bits(index_value, n_index_qubits)
    assert shot["state_0"][0] == state_register_bits[index_value]


def test_swapup_linear_register_statevector_branches_match_uncntrl_swaps() -> None:
    """Check all four coherent register SwapUp branches against uncontrolled swaps.

    A uniform index superposition gives each branch probability ``1 / 4``. The
    complete state must match the independently swapped reference up to global
    phase.
    """
    n_state_registers = 4
    n_index_qubits = ceil(log2(n_state_registers))
    res = _run_swapup_linear_register_statevector()

    states = Quest.extract_states_dict(res.results[0].entries)
    non_work_state, spec_qubits_dict = get_total_state_on_only_specified_registers(
        states, ["index", "state"]
    )
    index_qubits = spec_qubits_dict["index"]

    for index_value in range(n_state_registers):
        non_work_state.specified_qubits = index_qubits
        # Projection expects MSB-first bits; SwapUp indices are little-endian.
        projected = project_state_onto_bitstring(
            non_work_state,
            int_to_bits(index_value, n_index_qubits),
        )

        assert np.allclose(projected.probability, 1 / n_state_registers)
        assert_allclose_ignorephase(
            projected.state.state,
            _expected_register_branch_state(index_value),
        )


def test_swapup_linear_overload_resolves_nontrivial_targets_in_one_program() -> None:
    """Check the public ``swapup_linear`` overload with qubit and register targets.

    The program uses five qubits and five one-qubit state registers. It checks
    that both indices are unchanged and selected values move to
    ``state_qreg[0]`` and ``state_qregs[0]``.
    """
    n_index_qubits = 3
    n_state_items = 5

    @guppy
    @no_type_check
    def main() -> None:
        flat_index_qreg = array(qubit() for _ in range(comptime(n_index_qubits)))
        flat_state_qreg = array(qubit() for _ in range(comptime(n_state_items)))
        nested_index_qreg = array(qubit() for _ in range(comptime(n_index_qubits)))
        nested_state_qregs = array(
            array(qubit() for _ in range(comptime(1)))
            for _ in range(comptime(n_state_items))
        )

        x(flat_index_qreg[2])
        x(flat_state_qreg[4])
        x(nested_index_qreg[2])
        x(nested_state_qregs[4][0])

        swapup_linear(flat_index_qreg, flat_state_qreg)
        swapup_linear(nested_index_qreg, nested_state_qregs)

        result("flat_index", collect_measurements(measure_array(flat_index_qreg)))
        result("flat_state", collect_measurements(measure_array(flat_state_qreg)))
        result(
            "nested_index",
            collect_measurements(measure_array(nested_index_qreg)),
        )
        nested_state_0 = nested_state_qregs.take(0)
        result("nested_state_0", collect_measurements(measure_array(nested_state_0)))
        for register in range(1, len(nested_state_qregs)):
            discard_array(nested_state_qregs.take(register))
        nested_state_qregs.discard_all_taken()

    shot = (
        main.emulator(n_qubits=2 * (n_index_qubits + n_state_items))
        .with_seed(42)
        .with_shots(1)
        .run()
    ).collated_shots()[0]

    assert shot["flat_index"][0] == [False, False, True]
    assert shot["flat_state"][0] == [True, False, False, False, False]
    assert shot["nested_index"][0] == [False, False, True]
    assert shot["nested_state_0"][0] == [True]
