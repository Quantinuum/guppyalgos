"""Tests for pauli select."""

from math import ceil, log2
from collections.abc import Iterable

import pytest
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.emulator import EmulatorResult
from guppylang.std.angles import angle
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, h, ry
from selene_quest_plugin import SeleneQuestState
from selene_sim import Quest

from guppyalgos.algorithms.select.pauli_select import pauli_select_unary_iteration
from guppyalgos.utils import int_to_bits, qarray, transversal
from guppyalgos.primitives.pauli import pauli_to_gate
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    project_state_onto_bitstring,
    get_total_state_on_only_specified_registers,
)


from typing import cast, no_type_check


def run_uniform_pauli_select_unary_iteration(
    pauli_strings: zqp.Strings,
    n_index_qubits: int,
    n_state_qubits: int,
    ryangle: float,
) -> EmulatorResult:
    """Run pauli select using given Pauli strings with index as uniform superposition.

    State reg qubits are rotated so they do not lie on X or Z basis vectors.
    """
    select = pauli_select_unary_iteration(pauli_strings, n_state_qubits)

    @guppy
    @no_type_check
    def select_on_uniform_index() -> None:
        q = qarray(n_state_qubits)
        a = qarray(n_index_qubits)
        transversal(h, a)
        # act on some state not in the Z or X bases
        for i in range(n_state_qubits):
            ry(q[i], angle(ryangle))
        select(a, q)
        state_output("index", a)
        state_output("state", q)
        discard_array(a)
        discard_array(q)

    total_qubits = n_state_qubits + n_index_qubits + (n_index_qubits - 1)
    return select_on_uniform_index.emulator(n_qubits=total_qubits).with_seed(42).run()


@pytest.mark.parametrize("ryangle", [0.0, 0.23, 0.57, 1.0])
@pytest.mark.parametrize(
    ("pauli_list_str", "n_state_qubits"),
    [
        ("X0 X1 X3, X1 Y2 Z3, X3, Y2, Z1 X3", 4),
        ("X0, Y0, Z0", 1),
        ("X0 X1, Y0 Y1, Z0 Z1", 2),
        ("X0 X2, Y1, Z0 X1 Z2", 3),
        ("X0 X1 Y2 Z3 X4, Y0 Y2, Z1", 5),
    ],
)
def test_pauli_select_unary_iteration(
    pauli_list_str: str, n_state_qubits: int, ryangle: float
) -> None:
    """Test correct paulis are applied to index states.

    Run pauli select acting on a rotated state to ensure non-trivial aaction of the
    paulis, then assert that for each index bitstring, the projected state on the main
    register is the same as having applied the relevant pauli on a single register.
    """
    pauli_strings = zqp.Strings.from_str(pauli_list_str, n_state_qubits)
    n_index_qubits = ceil(log2(len(pauli_strings)))

    res = run_uniform_pauli_select_unary_iteration(
        pauli_strings, n_index_qubits, n_state_qubits, ryangle
    )
    states = Quest.extract_states_dict(res.results[0].entries)
    non_work_state, spec_qubits_dict = get_total_state_on_only_specified_registers(
        states, ["index", "state"]
    )
    index_qubits = spec_qubits_dict["index"]

    def make_solo_pauli_state(
        pauli_string: zqp.String, n_state_qubits: int
    ) -> SeleneQuestState:
        pauli_func = pauli_to_gate(pauli_string, n_state_qubits)

        @guppy
        @no_type_check
        def pauli_alone() -> None:
            q = qarray(n_state_qubits)
            # act on some state not in the Z or X bases
            for i in range(n_state_qubits):
                ry(q[i], angle(ryangle))
            pauli_func(q)
            state_output("state", q)
            discard_array(q)

        return Quest.extract_states_dict(
            pauli_alone.emulator(n_state_qubits).run().results[0].entries
        )["state"]

    # test that every index bitstring has had the correct pauli applied
    for i, pauli in enumerate(cast(Iterable[zqp.String], pauli_strings)):
        # get state attached to index bitstring
        non_work_state.specified_qubits = index_qubits  # project on index
        bitstring = int_to_bits(i, n_index_qubits)
        projindex = project_state_onto_bitstring(non_work_state, bitstring)

        # generate the pauli applied to a register by itself
        solo_pauli_state = make_solo_pauli_state(pauli, n_state_qubits)

        assert_allclose_ignorephase(projindex.state.state, solo_pauli_state.state)
