"""Tests for accumulator unary iteration primitives.

These tests verify:

- Correct behavior of the controlled_ops used as test inputs.
- Correct encoding of the index qubit register.
- Classical correctness of accumulator_unary_iteration for valid indices.
- Statevector correctness for original and guarded accumulator variants.
- Identity behavior of the guarded variant on out-of-range index branches.

Work qubits and the accumulator qubit are discarded by the primitives;
correctness is validated on the final index and state registers.
"""

from functools import cache

from guppylang.emulator import EmulatorInstance

from math import ceil, log2
from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import Function, array, output
from guppylang.std.debug import state_output
from guppylang.std.quantum import (
    angle,
    collect_measurements,
    cx,
    discard_array,
    h,
    measure_array,
    qubit,
    ry,
    toffoli,
    x,
)
from selene_sim import Quest

from guppyalgos.algorithms.select.accumulator.accumulator_unary_iteration import (
    accumulator_unary_iteration,
    guarded_accumulator_unary_iteration,
)
from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.primitives.state_preparation import uniform_state
from guppyalgos.utils import int_to_bits, qarray, transversal
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_total_state_on_only_specified_registers,
    project_state_onto_bitstring,
    switch_endianness,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def make_single_x_op(k: int, n_state_qubits: int) -> GuppyFunctionDefinition:
    """Return controlled_ops[k] = CX(control, state[k])."""

    @guppy
    @no_type_check
    def op(control: qubit, state: array[qubit, n_state_qubits]) -> None:
        cx(control, state[k])

    return op


@cache
def _expected_branch_state_emulator(n_state_qubits: int) -> EmulatorInstance:
    @guppy
    @no_type_check
    def branch(ldx: int, ryangle: float) -> None:
        state = qarray(n_state_qubits)

        for i in range(n_state_qubits):
            ry(state[i], angle(ryangle))

        for k in range(ldx):
            x(state[k])

        state_output("state", state)
        discard_array(state)

    return branch.emulator(n_state_qubits)


def expected_branch_state(ldx: int, n_state_qubits: int, ryangle: float) -> np.ndarray:
    """Return the ideal statevector after applying controlled_ops[0..ldx-1]."""
    emulator = _expected_branch_state_emulator(n_state_qubits)
    res = emulator.run(ldx=ldx, ryangle=ryangle)
    sv = Quest.extract_states_dict(res.results[0].entries)["state"].get_single_state()
    return switch_endianness(sv)


def make_cntrl_op_test_circuit(
    k: int,
    n_state: int,
) -> GuppyFunctionDefinition:
    """Return a circuit that tests controlled_ops[k]."""
    controlled_op = make_single_x_op(k, n_state)

    @guppy
    @no_type_check
    def main() -> None:
        ctrl = qubit()
        x(ctrl)

        state = qarray(n_state)
        controlled_op(ctrl, state)

        output("state", collect_measurements(measure_array(state)))
        output("ctrl", collect_measurements(measure_array(array(ctrl))))

    return main


# AND compute/uncompute primitives tested across accumulator tests.
AND_PRIMITIVES = [
    (toffoli, toffoli),
    (temp_and_compute, temp_and_uncompute),
]


# -----------------------------------------------------------------------------
# Test 1: controlled_ops correctness
# -----------------------------------------------------------------------------
def test_cntrl_ops_correct_qubit() -> None:
    """Verify that each controlled_op flips exactly the intended qubit."""
    n_state = 4

    for k in range(n_state):
        main = make_cntrl_op_test_circuit(k, n_state)
        shots = main.emulator(n_qubits=n_state + 1).with_seed(42).with_shots(1).run()

        measured = shots.collated_shots()[0]["state"][0]
        expected = [1 if i == k else 0 for i in range(n_state)]

        assert measured == expected


# -----------------------------------------------------------------------------
# Test 2: index encoding correctness
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("ldx", [0, 1, 2, 3])
def test_index_prep_encoding(ldx: int) -> None:
    """Verify the little-endian encoding of |l> for four index elements."""
    n_index_qubits = ceil(log2(4))
    idx_bits = int_to_bits(ldx, n_index_qubits)

    @guppy
    @no_type_check
    def main() -> None:
        idx = idx_bits
        index = qarray(n_index_qubits)

        for b in range(n_index_qubits):
            if idx[b]:
                x(index[b])

        output("index", collect_measurements(measure_array(index)))

    shots = main.emulator(n_qubits=n_index_qubits).with_seed(42).with_shots(1).run()

    assert shots.collated_shots()[0]["index"][0] == idx_bits


# -----------------------------------------------------------------------------
# Test 3: classical correctness of accumulator_unary_iteration
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(("comp_and_op", "uncomp_and_op"), AND_PRIMITIVES)
@pytest.mark.parametrize(
    ("n_index_elements", "ldx"),
    [(n, ldx) for n in [4, 8] for ldx in range(n)],
)
def test_accumulator_unary_iteration_basic(
    comp_and_op: GuppyFunctionDefinition,
    uncomp_and_op: GuppyFunctionDefinition,
    n_index_elements: int,
    ldx: int,
) -> None:
    """Check that controlled_ops[0..l-1] are applied when index = l."""
    n_index_qubits = ceil(log2(n_index_elements))
    n_state_qubits = n_index_elements
    idx_bits = int_to_bits(ldx, n_index_qubits)

    @guppy.comptime
    @no_type_check
    def build_ops() -> array[
        Function[[qubit, array[qubit, n_state_qubits]], None],
        n_index_elements,
    ]:
        return [make_single_x_op(k, n_state_qubits) for k in range(n_index_elements)]

    @guppy
    @no_type_check
    def main() -> None:
        idx = idx_bits
        index = qarray(n_index_qubits)

        for b in range(n_index_qubits):
            if idx[b]:
                x(index[b])

        state = qarray(n_state_qubits)
        ops = build_ops()

        accumulator_unary_iteration(ops, comp_and_op, uncomp_and_op, index, state)

        output("state", collect_measurements(measure_array(state)))
        discard_array(index)

    total_qubits = n_index_qubits + n_state_qubits + (n_index_qubits - 1) + 1
    shots = main.emulator(n_qubits=total_qubits).with_seed(42).with_shots(1).run()

    measured = shots.collated_shots()[0]["state"][0]
    expected = [1] * ldx + [0] * (n_state_qubits - ldx)

    assert measured == expected


# -----------------------------------------------------------------------------
# Test 4: statevector correctness of accumulator_unary_iteration
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(("comp_and_op", "uncomp_and_op"), AND_PRIMITIVES)
@pytest.mark.parametrize("n_index_elements", [4, 5])
@pytest.mark.parametrize("ryangle", [0.31, 0.7])
def test_accumulator_unary_iteration_statevector(
    comp_and_op: GuppyFunctionDefinition,
    uncomp_and_op: GuppyFunctionDefinition,
    n_index_elements: int,
    ryangle: float,
) -> None:
    """Verify each valid index branch against an independent ideal state."""
    n_index_qubits = ceil(log2(n_index_elements))
    n_state_qubits = n_index_elements

    @guppy.comptime
    @no_type_check
    def build_ops() -> array[
        Function[[qubit, array[qubit, n_state_qubits]], None],
        n_index_elements,
    ]:
        return [make_single_x_op(k, n_state_qubits) for k in range(n_index_elements)]

    uniform_over_n_elements_prep = uniform_state(n_index_elements)

    @guppy
    @no_type_check
    def main() -> None:
        index = qarray(n_index_qubits)
        uniform_over_n_elements_prep(index)

        state = qarray(n_state_qubits)
        for i in range(n_state_qubits):
            ry(state[i], angle(ryangle))

        ops = build_ops()
        accumulator_unary_iteration(ops, comp_and_op, uncomp_and_op, index, state)

        state_output("index", index)
        state_output("state", state)
        discard_array(index)
        discard_array(state)

    total_qubits = n_index_qubits + n_state_qubits + (n_index_qubits - 1) + 1
    res = main.emulator(n_qubits=total_qubits).run()

    states = Quest.extract_states_dict(res.results[0].entries)
    non_work_state, spec_qubits_dict = get_total_state_on_only_specified_registers(
        states, ["index", "state"]
    )
    index_qubits = spec_qubits_dict["index"]

    index_bitstrings = [
        int_to_bits(i, n_index_qubits) for i in range(2**n_index_qubits)
    ]

    for ldx, bitstring in enumerate(index_bitstrings):
        non_work_state.specified_qubits = index_qubits
        projindex = project_state_onto_bitstring(non_work_state, bitstring)

        if ldx < n_index_elements:
            assert np.allclose(projindex.probability, 1 / n_index_elements)

            exp_vec = expected_branch_state(ldx, n_state_qubits, ryangle)
            assert_allclose_ignorephase(projindex.state.state, exp_vec)
        else:
            assert np.allclose(projindex.probability, 0.0)


# -----------------------------------------------------------------------------
# Test 5: statevector correctness of guarded_accumulator_unary_iteration
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(("comp_and_op", "uncomp_and_op"), AND_PRIMITIVES)
@pytest.mark.parametrize("n_index_elements", [3, 4])
@pytest.mark.parametrize("ryangle", [0.31, 0.7])
def test_guarded_accumulator_unary_iteration_statevector(
    comp_and_op: GuppyFunctionDefinition,
    uncomp_and_op: GuppyFunctionDefinition,
    n_index_elements: int,
    ryangle: float,
) -> None:
    """Verify valid branches and trivial out-of-range guarded branches."""
    n_index_qubits = ceil(log2(n_index_elements))
    n_state_qubits = n_index_elements

    @guppy.comptime
    @no_type_check
    def build_ops() -> array[
        Function[[qubit, array[qubit, n_state_qubits]], None],
        n_index_elements,
    ]:
        return [make_single_x_op(k, n_state_qubits) for k in range(n_index_elements)]

    @guppy
    @no_type_check
    def main() -> None:
        index = qarray(n_index_qubits)
        transversal(h, index)

        state = qarray(n_state_qubits)
        for i in range(n_state_qubits):
            ry(state[i], angle(ryangle))

        ops = build_ops()
        guarded_accumulator_unary_iteration(
            ops, comp_and_op, uncomp_and_op, index, state
        )

        state_output("index", index)
        state_output("state", state)
        discard_array(index)
        discard_array(state)

    total_qubits = n_index_qubits + n_state_qubits + (n_index_qubits - 1) + 1
    res = main.emulator(n_qubits=total_qubits).run()

    states = Quest.extract_states_dict(res.results[0].entries)
    non_work_state, spec_qubits_dict = get_total_state_on_only_specified_registers(
        states, ["index", "state"]
    )
    index_qubits = spec_qubits_dict["index"]

    index_bitstrings = [
        int_to_bits(i, n_index_qubits) for i in range(2**n_index_qubits)
    ]

    for ldx, bitstring in enumerate(index_bitstrings):
        non_work_state.specified_qubits = index_qubits
        projindex = project_state_onto_bitstring(non_work_state, bitstring)

        assert np.allclose(projindex.probability, 1 / (2**n_index_qubits))

        if ldx < n_index_elements:
            exp_vec = expected_branch_state(ldx, n_state_qubits, ryangle)
        else:
            exp_vec = expected_branch_state(0, n_state_qubits, ryangle)

        assert_allclose_ignorephase(projindex.state.state, exp_vec)
