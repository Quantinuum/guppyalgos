"""Tests for the qrom_unary_iteration function."""

from math import ceil, log2
from typing import no_type_check

import numpy as np
import pytest
from guppylang import array
from guppylang.decorator import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.emulator import EmulatorResult
from guppylang.std.builtins import output
from guppylang.std.debug import state_output
from guppylang.std.quantum import (
    collect_measurements,
    cx,
    discard_array,
    h,
    measure_array,
    qubit,
    toffoli,
    x,
)
from selene_sim import Quest

from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.primitives.subroutines.fanout import fanout_measurement_parity
from guppyalgos.algorithms.select.qrom import qrom_unary_iteration
from guppyalgos.algorithms.select.qrom import qrom_measure_uncompute
from guppyalgos.algorithms.select.select_unary_iteration import (
    get_index_bools,
    msb_diff_depths,
)
from guppyalgos.utils import int_to_bits, qarray, transversal
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_total_state_on_only_specified_registers,
    project_state_onto_bitstring,
)


def test_qrom_msb_diff_depths_follow_little_endian_order() -> None:
    """Check the most-significant differing bits are found from the array end."""
    index_bools = get_index_bools(3, 6, int_to_bits)

    assert index_bools == [
        [False, False, False],
        [True, False, False],
        [False, True, False],
        [True, True, False],
        [False, False, True],
        [True, False, True],
    ]
    # Scan from physical position 2 toward 0. The values are cascade depths,
    # so depth 0 is position 2 and depth 2 is position 0.
    assert msb_diff_depths(index_bools) == [0, 2, 1, 2, 0, 2]


def qrom_unary_iteration_test(
    n_index_elements: int,
    idxs: list[int],
    compute_temp_and: GuppyFunctionDefinition,
    uncompute_temp_and: GuppyFunctionDefinition,
) -> None:
    """Validate qrom_unary_iteration for a given index.

    Constructs a QROM where data[j] is the n_state_qubits-bit binary of j.
    It prepares index i on index_qreg, applies the QROM, and asserts that
    the measured state_qreg equals data_input[i].

    The test is exercised with two AND implementations:
    - Toffoli-only (compute and uncompute with toffoli).
    - 4T compute (compute_temp_and) with measurement-based uncomp (uncompute_temp_and).
    - It could be further extended to include other AND implementations.

    - It also tests cases where n_index_elements < 2^n_index_qubits.

    Args:
        n_index_elements: Number of index qubits (and state bits per entry).
        idxs: Indices to load and verify.
        compute_temp_and: AND compute primitive used inside the QROM gadget.
        uncompute_temp_and: AND uncompute primitive used inside the QROM gadget.

    """
    n_index_qubits = ceil(log2(n_index_elements))
    n_state_qubits = n_index_qubits

    # Shift the payloads so address zero also performs a non-trivial operation.
    data_input = [
        int_to_bits((i + 1) % (2**n_state_qubits), n_state_qubits)
        for i in range(n_index_elements)
    ]

    qrom = qrom_unary_iteration(
        data_input, comp_and_op=compute_temp_and, uncomp_and_op=uncompute_temp_and
    )

    @guppy
    @no_type_check
    def main(bits: array[bool, n_state_qubits]) -> None:
        index_qreg = qarray(n_index_qubits)
        state_qreg = qarray(n_state_qubits)

        for bit in range(n_index_qubits):
            if bits[bit]:
                x(index_qreg[bit])

        qrom(index_qreg, state_qreg)

        output("index", collect_measurements(measure_array(index_qreg)))
        output("state", collect_measurements(measure_array(state_qreg)))

    total_qubits = n_state_qubits + n_index_qubits + (n_index_qubits - 1)
    emulator = main.emulator(total_qubits)
    for i in idxs:
        input_bits = int_to_bits(i, n_state_qubits)
        my_shots = emulator.run(bits=input_bits)
        assert my_shots.collated_shots()[0]["state"][0] == data_input[i]


@pytest.mark.parametrize(
    ("n_index_elements"),
    [2**3, 2**4],
)
def test_qrom_unary_iteration_single_targ_toffoli(n_index_elements: int) -> None:
    """Test qrom_unary_iteration with toffoli-based AND."""
    qrom_unary_iteration_test(
        n_index_elements, list(range(n_index_elements)), toffoli, toffoli
    )


@pytest.mark.parametrize(
    ("n_index_elements"),
    [2**3, 2**4],
)
def test_qrom_unary_iteration_single_targ_google(n_index_elements: int) -> None:
    """Test qrom_unary_iteration with 4T compute and measurement-based uncompute AND."""
    qrom_unary_iteration_test(
        n_index_elements,
        list(range(n_index_elements)),
        temp_and_compute,
        temp_and_uncompute,
    )


@pytest.mark.parametrize(
    ("n_index_elements"),
    [3, 6, 13],
)
def test_qrom_unary_iteration_single_targ_google_shorter(n_index_elements: int) -> None:
    """Test qrom_unary_iteration with 4T compute and measurement-based uncompute AND.

    Where the index elements are less than 2^N.
    """
    qrom_unary_iteration_test(
        n_index_elements,
        list(range(n_index_elements)),
        temp_and_compute,
        temp_and_uncompute,
    )


def test_qrom_unary_iteration_measurement_parity_fanout() -> None:
    """Load QROM data using parity-based measurement fanout."""
    n_index_qubits = 2
    n_state_qubits = 4
    data_input = [
        [False, False, False, False],
        [True, False, True, True],
        [False, True, True, False],
        [True, True, True, True],
    ]
    qrom = qrom_unary_iteration(
        data_input,
        fanout_op=fanout_measurement_parity,
    )

    @guppy
    @no_type_check
    def main(bits: array[bool, n_index_qubits]) -> None:
        index_qreg = qarray(n_index_qubits)
        state_qreg = qarray(n_state_qubits)

        for bit in range(n_index_qubits):
            if bits[bit]:
                x(index_qreg[bit])

        qrom(index_qreg, state_qreg)

        output("index", collect_measurements(measure_array(index_qreg)))
        output("state", collect_measurements(measure_array(state_qreg)))

    emulator = main.emulator(n_qubits=9).with_seed(42).with_shots(1)
    for index, expected in enumerate(data_input):
        result = emulator.run(bits=int_to_bits(index, n_index_qubits)).collated_shots()[
            0
        ]
        assert result["state"][0] == expected


def qrom_unary_iteration_test_statevector(
    n_index_elements: int,
    compute_temp_and: GuppyFunctionDefinition,
    uncompute_temp_and: GuppyFunctionDefinition,
) -> EmulatorResult:
    """Validate qrom_unary_iteration for a given index.

    Constructs a QROM where data[j] is the n_state_qubits-bit binary of j.
    Prepares the uniform state on the index qubits, and produces the output statevector.

    Args:
        n_index_elements: Number of index qubits (and state bits per entry).
        compute_temp_and: AND compute primitive used inside the QROM gadget.
        uncompute_temp_and: AND uncompute primitive used inside the QROM gadget.

    """
    n_index_qubits = ceil(log2(n_index_elements))
    n_state_qubits = n_index_qubits

    data_input = [int_to_bits(i, n_state_qubits) for i in range(n_index_elements)]

    qrom = qrom_unary_iteration(
        data_input, comp_and_op=compute_temp_and, uncomp_and_op=uncompute_temp_and
    )

    def test_input() -> GuppyFunctionDefinition:
        @guppy
        @no_type_check
        def main() -> None:
            index_qreg = qarray(n_index_qubits)
            state_qreg = qarray(n_state_qubits)

            for bit in range(n_index_qubits):
                h(index_qreg[bit])

            qrom(index_qreg, state_qreg)

            state_output("index", index_qreg)
            state_output("state", state_qreg)
            discard_array(index_qreg)
            discard_array(state_qreg)

        return main

    main_i = test_input()
    total_qubits = n_state_qubits + n_index_qubits + (n_index_qubits - 1)
    result = main_i.emulator(n_qubits=total_qubits).with_seed(42).run()
    return result


@pytest.mark.parametrize(("n_index_qubits"), [3, 4])
def test_qrom_unary_iteration_statevector(n_index_qubits: int) -> None:
    """Test unary iteration applied to a uniform superposition.

    Checks that each index state is matched with the same bitstring on the output qubits
    This also verifies that no relative phases have been introduced between the branches
    """
    n_index_elements = 2**n_index_qubits
    n_state_qubits = n_index_qubits
    res = qrom_unary_iteration_test_statevector(
        n_index_elements, temp_and_compute, temp_and_uncompute
    )
    states = Quest.extract_states_dict(res.results[0].entries)
    non_work_state, spec_qubits_dict = get_total_state_on_only_specified_registers(
        states, ["index", "state"]
    )
    index_qubits = spec_qubits_dict["index"]
    state_qubits = spec_qubits_dict["state"]
    # test that every index bitstring matches the bitstring on the state qubits
    data_input = [int_to_bits(i, n_state_qubits) for i in range(n_index_elements)]
    for bitstring in data_input:
        non_work_state.specified_qubits = index_qubits  # project on index
        projindex = project_state_onto_bitstring(non_work_state, bitstring)
        assert not np.allclose(projindex.probability, 0)
        non_work_state.specified_qubits = state_qubits  # project on state
        projstate = project_state_onto_bitstring(non_work_state, bitstring)
        np.testing.assert_allclose(projindex.state.state, projstate.state.state)


@pytest.mark.parametrize(("n_index_qubits"), [2, 3, 4])
@pytest.mark.parametrize(
    ("compute_and", "uncompute_and"),
    [
        (toffoli, toffoli),
        (temp_and_compute, temp_and_uncompute),
    ],
)
def test_qrom_inversion(
    n_index_qubits: int,
    compute_and: GuppyFunctionDefinition[[qubit, qubit, qubit], None],
    uncompute_and: GuppyFunctionDefinition[[qubit, qubit, qubit], None],
) -> None:
    """Test qrom using the fact that data index loading is the same as register cx."""
    n_index_elements = 2**n_index_qubits
    n_state_qubits = n_index_qubits

    data_input = [int_to_bits(i, n_state_qubits) for i in range(n_index_elements)]

    qrom = qrom_unary_iteration(
        data_input, comp_and_op=compute_and, uncomp_and_op=uncompute_and
    )

    @guppy
    @no_type_check
    def main() -> None:
        index_qreg = qarray(n_index_qubits)
        state_qreg = qarray(n_state_qubits)

        for bit in range(n_index_qubits):
            h(index_qreg[bit])

        qrom(index_qreg, state_qreg)

        # qrom with this data input is equivalent to register-wise cx, so invert
        for i in range(n_index_qubits):
            cx(index_qreg[i], state_qreg[i])

        for bit in range(n_index_qubits):
            h(index_qreg[bit])

        state_output("index", index_qreg)
        state_output("state", state_qreg)
        discard_array(index_qreg)
        discard_array(state_qreg)

    total_qubits = n_state_qubits + n_index_qubits + (n_index_qubits - 1)
    result = main.emulator(n_qubits=total_qubits).run()
    states = Quest.extract_states_dict(result.results[0].entries)
    zero_state = np.zeros(2**n_state_qubits)
    zero_state[0] = 1
    # assert 0, ignore global phase for now
    assert_allclose_ignorephase(states["index"].get_single_state(), zero_state)
    assert_allclose_ignorephase(states["state"].get_single_state(), zero_state)


@pytest.mark.parametrize(("n_index_qubits"), [3, 4])
def test_qrom_measurement_uncompute(n_index_qubits: int) -> None:
    """Test measurement based qrom uncompute by asserting we return to the 0 state."""
    n_index_elements = 2**n_index_qubits
    n_state_qubits = n_index_qubits
    data = [int_to_bits(i, n_state_qubits) for i in range(n_index_elements)]

    qrom = qrom_unary_iteration(data)
    qrom_uncomp = qrom_measure_uncompute(data)

    @guppy
    @no_type_check
    def main() -> None:
        index = qarray(n_index_qubits)
        state = qarray(n_state_qubits)
        transversal(h, index)
        qrom(index, state)
        qrom_uncomp(index, state)
        transversal(h, index)
        state_output("output", index)
        discard_array(index)

    zero_state = np.zeros(2**n_index_qubits)
    zero_state[0] = 1
    n_shots = 10
    total_qubits = n_state_qubits + n_index_qubits + (n_index_qubits - 1)
    em_results = main.emulator(total_qubits).with_shots(n_shots).run()
    for res in em_results.results:
        state = Quest.extract_states_dict(res.entries)
        np.testing.assert_allclose(state["output"].get_single_state(), zero_state)
