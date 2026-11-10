"""Pauli select primitives."""

from guppyalgos.algorithms.select import build_select_unary_from_data


import zixy.qubit.pauli as zqp
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import qubit

from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and,
    uncompute_and,
)
from guppyalgos.primitives.pauli import pauli_to_cntrl_gate


def pauli_select_unary_iteration[n_i_q: nat, n_s_q: nat](
    pauli_strings: zqp.Strings,
    n_state_qubits: int,
    comp_and_op: GuppyFunctionDefinition[[qubit, qubit, qubit], None] = temp_and,
    uncomp_and_op: GuppyFunctionDefinition[[qubit, qubit, qubit], None] = uncompute_and,
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
        lambda p: pauli_to_cntrl_gate(p, n_state_qubits),
        comp_and_op,
        uncomp_and_op,
    )
