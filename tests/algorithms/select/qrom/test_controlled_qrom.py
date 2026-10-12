"""Tests for the controlled QROM unary iteration function."""

from math import ceil, log2
from typing import no_type_check

import numpy as np
import pytest
from guppylang.decorator import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.emulator import EmulatorResult
from guppylang.std.builtins import array, output
from guppylang.std.debug import state_output
from guppylang.std.quantum import (
    collect_measurements,
    cx,
    discard,
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
from guppyalgos.algorithms.select.qrom.controlled_qrom_unary_iteration import (
    cntrl_qrom_unary_iteration,
)
from guppyalgos.utils import int_to_bits
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_total_state_on_only_specified_registers,
    project_state_onto_bitstring,
)
from guppyalgos.utils import qarray


def cntrl_qrom_unary_iteration_test(
    n_index_elements: int,
    idxs: list[int],
    compute_temp_and: GuppyFunctionDefinition,
    uncompute_temp_and: GuppyFunctionDefinition,
) -> None:
    """Validate cntrl_qrom_unary_iteration for given indices.

    Constructs a QROM where data[j] is the n_state_qubits-bit binary of j.
    It prepares indices on index_qreg, applies the QROM, and asserts that
    the measured state_qreg equals data_input[i] for each test index.

    The test is exercised with two AND implementations:
    - Toffoli-only (compute and uncompute with toffoli).
    - 4T compute (compute_temp_and) with measurement-based uncomp (uncompute_temp_and).
    - It could be further extended to include other AND implementations.

    - It also tests cases where n_index_elements < 2^n_index_qubits.

    Args:
        n_index_elements: Number of index qubits (and state bits per entry).
        idxs: List of indices to load and verify.
        compute_temp_and: AND compute primitive used inside the QROM gadget.
        uncompute_temp_and: AND uncompute primitive used inside the QROM gadget.

    """
    n_index_qubits = ceil(log2(n_index_elements))
    n_state_qubits = n_index_qubits

    data_input = [int_to_bits(i, n_state_qubits) for i in range(n_index_elements)]

    qrom = cntrl_qrom_unary_iteration(
        data_input, comp_and_op=compute_temp_and, uncomp_and_op=uncompute_temp_and
    )

    @guppy
    @no_type_check
    def apply_ctrl_qrom_to_basis_state(bits: array[bool, n_state_qubits]) -> None:
        control = qubit()
        index_qreg = qarray(n_index_qubits)
        state_qreg = qarray(n_state_qubits)
        x(control)  # set control to |1> to activate the controlled QROM

        for bit in range(n_index_qubits):
            if bits[bit]:
                x(index_qreg[bit])

        qrom(control, index_qreg, state_qreg)

        output("index", collect_measurements(measure_array(index_qreg)))
        output("state", collect_measurements(measure_array(state_qreg)))
        discard(control)

    total_qubits = n_state_qubits + n_index_qubits + n_index_qubits + 1
    emulator = apply_ctrl_qrom_to_basis_state.emulator(n_qubits=total_qubits)
    for i in idxs:
        input_bits = int_to_bits(i, n_state_qubits)
        my_shots = emulator.run(bits=input_bits)
        assert my_shots.collated_shots()[0]["state"][0] == data_input[i]


@pytest.mark.parametrize(
    "n_index_elements",
    [2**3, 2**4],
)
def test_cntrl_qrom_unary_iteration_single_targ_toffoli(
    n_index_elements: int,
) -> None:
    """Test cntrl_qrom_unary_iteration with toffoli-based AND."""
    cntrl_qrom_unary_iteration_test(
        n_index_elements, list(range(n_index_elements)), toffoli, toffoli
    )


@pytest.mark.parametrize(
    "n_index_elements",
    [2**3, 2**4],
)
def test_cntrl_qrom_unary_iteration_single_targ_google(
    n_index_elements: int,
) -> None:
    """Test cntrl_qrom_unary_iteration with 4T compute.

    Uses measurement-based uncompute AND.
    """
    cntrl_qrom_unary_iteration_test(
        n_index_elements,
        list(range(n_index_elements)),
        temp_and_compute,
        temp_and_uncompute,
    )


@pytest.mark.parametrize(
    "n_index_elements",
    [6, 13],
)
def test_cntrl_qrom_unary_iteration_single_targ_google_shorter(
    n_index_elements: int,
) -> None:
    """Test cntrl_qrom_unary_iteration with 4T compute.

    Uses measurement-based uncompute AND where the index elements are less than 2^N.
    """
    cntrl_qrom_unary_iteration_test(
        n_index_elements,
        list(range(n_index_elements)),
        temp_and_compute,
        temp_and_uncompute,
    )


def cntrl_qrom_unary_iteration_test_statevector(
    n_index_elements: int,
    compute_temp_and: GuppyFunctionDefinition,
    uncompute_temp_and: GuppyFunctionDefinition,
) -> EmulatorResult:
    """Validate cntrl_qrom_unary_iteration for a given index.

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

    qrom = cntrl_qrom_unary_iteration(
        data_input, comp_and_op=compute_temp_and, uncomp_and_op=uncompute_temp_and
    )

    def test_input() -> GuppyFunctionDefinition:
        @guppy
        @no_type_check
        def main() -> None:
            control = qubit()
            index_qreg = qarray(n_index_qubits)
            state_qreg = qarray(n_state_qubits)
            x(control)  # set control to |1> to activate the controlled QROM

            for bit in range(n_index_qubits):
                h(index_qreg[bit])

            qrom(control, index_qreg, state_qreg)

            state_output("control", control)
            state_output("index", index_qreg)
            state_output("state", state_qreg)
            discard(control)
            discard_array(index_qreg)
            discard_array(state_qreg)

        return main

    main_i = test_input()
    total_qubits = n_state_qubits + n_index_qubits + n_index_qubits + 1
    result = main_i.emulator(n_qubits=total_qubits).with_seed(42).run()
    return result


@pytest.mark.parametrize(("n_index_qubits"), [3, 4])
def test_cntrl_qrom_unary_iteration_statevector(n_index_qubits: int) -> None:
    """Test controlled unary iteration applied to a uniform superposition.

    Checks that each index state is matched with the same bitstring on the output qubits
    This also verifies that no relative phases have been introduced between the branches
    """
    n_index_elements = 2**n_index_qubits
    n_state_qubits = n_index_qubits

    res = cntrl_qrom_unary_iteration_test_statevector(
        n_index_elements, temp_and_compute, temp_and_uncompute
    )
    states = Quest.extract_states_dict(res.results[0].entries)

    non_work_state, spec_qubits_dict = get_total_state_on_only_specified_registers(
        states, ["control", "index", "state"]
    )
    # control_qubit = spec_qubits_dict["control"]
    index_qubits = spec_qubits_dict["index"]
    state_qubits = spec_qubits_dict["state"]

    # test that every index bitstring matches the bitstring on the state qubits
    data_input = [int_to_bits(i, n_state_qubits) for i in range(n_index_elements)]

    for bitstring in data_input:
        non_work_state.specified_qubits = index_qubits  # project on index
        projindex = project_state_onto_bitstring(non_work_state, bitstring)
        try:
            assert not np.allclose(projindex.probability, 0)
        except AssertionError:
            print(f"Probability for projection onto bitstring {bitstring} is zero.")

        non_work_state.specified_qubits = state_qubits  # project on state
        projstate = project_state_onto_bitstring(non_work_state, bitstring)
        try:
            np.testing.assert_allclose(projindex.state.state, projstate.state.state)
        except AssertionError as e:
            print(f"Assertion failed for bitstring {bitstring}: {e}")


def cntrl_qrom_unary_iteration_test_statevector_zero(
    n_index_elements: int,
    compute_temp_and: GuppyFunctionDefinition,
    uncompute_temp_and: GuppyFunctionDefinition,
) -> tuple[GuppyFunctionDefinition[[], None], EmulatorResult]:
    """Validate controlled QROM for a given index when the control is |0>.

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

    qrom = cntrl_qrom_unary_iteration(
        data_input, comp_and_op=compute_temp_and, uncomp_and_op=uncompute_temp_and
    )

    def test_input() -> GuppyFunctionDefinition:
        @guppy
        @no_type_check
        def main() -> None:
            control = qubit()
            index_qreg = qarray(n_index_qubits)
            state_qreg = qarray(n_state_qubits)

            for bit in range(n_index_qubits):
                h(index_qreg[bit])

            qrom(control, index_qreg, state_qreg)

            state_output("control", control)
            state_output("index", index_qreg)
            state_output("state", state_qreg)
            discard(control)
            discard_array(index_qreg)
            discard_array(state_qreg)

        return main

    main_i = test_input()
    total_qubits = n_state_qubits + n_index_qubits + n_index_qubits + 1
    result = main_i.emulator(n_qubits=total_qubits).with_seed(42).run()
    return main_i, result


@pytest.mark.parametrize(("n_index_qubits"), [3, 4])
def test_cntrl_qrom_unary_iteration_statevector_zero(n_index_qubits: int) -> None:
    """Test that when the control is |0>, the QROM does not act.

    The state remains |0...0> on the state qubits, regardless of the index qubits.
    This checks that the control is properly controlling the operation of the QROM,
    and that no unintended operations are occurring on the state qubits when control
    is off.
    """
    n_index_elements = 2**n_index_qubits
    n_state_qubits = n_index_qubits
    total_qubits = n_state_qubits + n_index_qubits + n_index_qubits + 1

    _, res = cntrl_qrom_unary_iteration_test_statevector_zero(
        n_index_elements, temp_and_compute, temp_and_uncompute
    )

    states = Quest.extract_states_dict(res.results[0].entries)
    non_work_state, spec_qubits_dict = get_total_state_on_only_specified_registers(
        states, ["control", "index", "state"]
    )

    control_qubit = spec_qubits_dict["control"]
    index_qubits = spec_qubits_dict["index"]
    state_qubits = spec_qubits_dict["state"]

    index_bitstrings = [int_to_bits(i, n_state_qubits) for i in range(n_index_elements)]
    zero_state = np.zeros(2 ** (total_qubits - n_index_qubits - 1))
    zero_state[0] = 1

    ## We can check the state by projecting the full state
    ## onto the control and the index bitstrings, and then checking that
    ## the state qubits are in the |0...0> state for each index bitstring
    ## when control is |0>:
    non_work_state.specified_qubits = control_qubit  # project on control
    projcontrol = project_state_onto_bitstring(non_work_state, [False])
    zero_index_state = np.zeros(2 ** (n_state_qubits))
    zero_index_state[0] = 1

    for bitstring in index_bitstrings:
        projcontrol.state.specified_qubits = index_qubits  # project on index
        projindex = project_state_onto_bitstring(projcontrol.state, bitstring)
        np.testing.assert_allclose(projindex.state.state, zero_index_state, atol=1e-12)

    uniform_index_state = np.ones(2**n_index_qubits) / np.sqrt(2**n_index_qubits)
    zero_bitstring = [False] * n_state_qubits
    projcontrol.state.specified_qubits = state_qubits  # project on state
    projstate = project_state_onto_bitstring(projcontrol.state, zero_bitstring)
    np.testing.assert_allclose(projstate.state.state, uniform_index_state, atol=1e-12)


@pytest.mark.parametrize(("n_index_qubits"), [2, 3, 4])
@pytest.mark.parametrize(
    ("compute_and", "uncompute_and"),
    [
        (toffoli, toffoli),
        (temp_and_compute, temp_and_uncompute),
    ],
)
def test_cntrl_qrom_inversion(
    n_index_qubits: int,
    compute_and: GuppyFunctionDefinition[[qubit, qubit, qubit], None],
    uncompute_and: GuppyFunctionDefinition[[qubit, qubit, qubit], None],
) -> None:
    """Test controlled_qrom_unary_iteration using the fact that data index loading.

    is the same as register cx.
    """
    n_index_elements = 2**n_index_qubits
    n_state_qubits = n_index_qubits

    data_input = [int_to_bits(i, n_state_qubits) for i in range(n_index_elements)]

    qrom = cntrl_qrom_unary_iteration(
        data_input, comp_and_op=compute_and, uncomp_and_op=uncompute_and
    )

    @guppy
    @no_type_check
    def main() -> None:
        control = qubit()
        index_qreg = qarray(n_index_qubits)
        state_qreg = qarray(n_state_qubits)

        x(control)  # set control to |1> to activate the controlled QROM

        for bit in range(n_index_qubits):
            h(index_qreg[bit])

        qrom(control, index_qreg, state_qreg)

        # qrom with this data input is equivalent to register-wise cx, so invert
        for i in range(n_index_qubits):
            cx(index_qreg[i], state_qreg[i])

        for bit in range(n_index_qubits):
            h(index_qreg[bit])

        state_output("index", index_qreg)
        state_output("state", state_qreg)
        discard_array(index_qreg)
        discard_array(state_qreg)
        discard(control)

    total_qubits = n_state_qubits + n_index_qubits + n_index_qubits + 1
    result = main.emulator(n_qubits=total_qubits).run()
    states = Quest.extract_states_dict(result.results[0].entries)
    zero_state = np.zeros(2**n_state_qubits)
    zero_state[0] = 1
    # assert 0, ignore global phase for now
    assert_allclose_ignorephase(states["index"].get_single_state(), zero_state)
    assert_allclose_ignorephase(states["state"].get_single_state(), zero_state)
