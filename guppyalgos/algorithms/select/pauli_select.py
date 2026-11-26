"""Pauli select primitives."""

from typing import no_type_check

import zixy.qubit.pauli as zqp
from guppylang import comptime, guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import Unitary, array, control, nat
from guppylang.std.quantum import qubit

from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.primitives.pauli import pauli_to_gate
from guppyalgos.primitives.utilities.gates import apply_phase

from .select_unary_iteration import build_select_unary_from_data


def _select_pauli_term[n: nat](
    pauli_string: zqp.String, size: int, phase: float = 0.0
) -> Unitary[[qubit, array[qubit, n]], None]:
    """Apply a Pauli and a phase in half-turns on the selected branch."""
    pauli = pauli_to_gate(pauli_string, size)

    @guppy(unitary=True)
    @no_type_check
    def select_term(flag: qubit, qs: array[qubit, comptime(size)]) -> None:
        with control(flag):
            pauli(qs)
        apply_phase(flag, angle(comptime(phase)))

    return select_term


def pauli_select_unary_iteration[n_i_q: nat, n_s_q: nat](
    pauli_strings: zqp.Strings,
    n_state_qubits: int,
    comp_and_op: GuppyFunctionDefinition[
        [qubit, qubit, qubit], None
    ] = temp_and_compute,
    uncomp_and_op: GuppyFunctionDefinition[
        [qubit, qubit, qubit], None
    ] = temp_and_uncompute,
) -> GuppyFunctionDefinition[[array[qubit, n_i_q], array[qubit, n_s_q]], None]:
    """Pauli select based on unary iteration.

    Applies the ith element of the pauli list controlled on the bitstring i in the index
    register, with inputs allowed for varying AND (un)computation used in the unary
    iteration. See select_unary_iteration for details of the iteration procedure.

    Args:
        pauli_strings (zqp.Strings): Pauli strings where `pauli_strings[i]` is applied
            on the state register controlled on the bitstring `i` in the index register
        n_state_qubits: number of qubits in the register where the Pauli strings are
            applied
        comp_and_op: Function to compute temporary AND operations.
        uncomp_and_op: Function to uncompute temporary AND operations.

    Returns:
        GuppyFunctionDefinition: Pauli select acting on index and state registers.

    """
    return build_select_unary_from_data(
        pauli_strings,
        lambda term: _select_pauli_term(term, n_state_qubits),
        comp_and_op,
        uncomp_and_op,
    )
