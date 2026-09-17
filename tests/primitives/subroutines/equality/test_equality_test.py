"""Tests for equality-test primitives."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, nat
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard, discard_array, h, qubit
from selene_sim import Quest

from guppyalgos.primitives.subroutines.equality import equality_test
from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx_single_ancilla
from guppyalgos.utils import int_to_bits, qarray, transversal
from guppyalgos.testing import (
    get_total_state_on_only_specified_registers,
    project_state_onto_bitstring,
)


@dataclass(frozen=True)
class CnxMethod[n_controls: nat]:
    """Wrapper for exact cnx methods used in equality tests."""

    name: str
    cnx_func: GuppyFunctionDefinition[[array[qubit, n_controls], qubit], None]
    ancilla_func: Callable[[int], int]


EXACT_CNX_METHODS = [
    CnxMethod("cnx_single_ancilla", cnx_single_ancilla, lambda _: 1),
]


@pytest.mark.parametrize(
    "cnx_method", EXACT_CNX_METHODS, ids=lambda method: method.name
)
@pytest.mark.parametrize("n_register_qubits", [1, 3, 5])
def test_equality_test_flag_marks_equal_pairs(
    cnx_method: CnxMethod, n_register_qubits: int
) -> None:
    """Check the equality flag on all basis-state pairs from one superposition run.

    The circuit is applied once to Hadamard-prepared ``lhs`` and ``rhs``
    registers. The resulting state is then projected onto every computational
    basis choice for ``lhs`` and ``rhs`` using the little-endian bit ordering
    exercised by the simulator helpers. For each projected pair, the remaining
    flag qubit must be ``|1>`` exactly when the two projected register values
    are equal and ``|0>`` otherwise.

    """
    cnx_func = cnx_method.cnx_func
    n_ancillas = cnx_method.ancilla_func(n_register_qubits)

    @guppy
    @no_type_check
    def cnx_box(control: array[qubit, n_register_qubits], target: qubit) -> None:
        cnx_func(control, target)

    @guppy
    @no_type_check
    def main() -> None:
        lhs = qarray(n_register_qubits)
        rhs = qarray(n_register_qubits)
        flag = qubit()

        transversal(h, lhs)
        transversal(h, rhs)
        equality_test(cnx_box, lhs, rhs, flag)

        state_output("lhs", lhs)
        state_output("rhs", rhs)
        state_output("flag", flag)
        discard_array(lhs)
        discard_array(rhs)
        discard(flag)

    res = main.emulator(2 * n_register_qubits + 1 + n_ancillas).run()
    states = Quest.extract_states_dict(res.results[0].entries)
    total_state, specified_qubits = get_total_state_on_only_specified_registers(
        states, ["lhs", "rhs", "flag"]
    )

    for lhs_value in range(2**n_register_qubits):
        lhs_bits = int_to_bits(lhs_value, n_register_qubits)
        total_state.specified_qubits = specified_qubits["lhs"]
        lhs_projected = project_state_onto_bitstring(
            total_state,
            lhs_bits,
            new_specified_qubits=specified_qubits["rhs"],
        )
        np.testing.assert_allclose(lhs_projected.probability, 1 / 2**n_register_qubits)

        for rhs_value in range(2**n_register_qubits):
            rhs_bits = int_to_bits(rhs_value, n_register_qubits)
            lhs_projected.state.specified_qubits = specified_qubits["rhs"]
            rhs_projected = project_state_onto_bitstring(
                lhs_projected.state,
                rhs_bits,
                new_specified_qubits=specified_qubits["flag"],
            )
            np.testing.assert_allclose(
                rhs_projected.probability, 1 / 2**n_register_qubits
            )
            flag_state = rhs_projected.state.state
            expected_flag_state = [0, 1] if lhs_value == rhs_value else [1, 0]
            np.testing.assert_allclose(flag_state, expected_flag_state)
