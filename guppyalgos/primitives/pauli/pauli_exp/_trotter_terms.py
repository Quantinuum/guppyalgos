"""Private Pauli term adapters for the existing Trotter callback contracts."""

from __future__ import annotations

from typing import no_type_check

import zixy.qubit.pauli as zqp
from guppylang import comptime, guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import array, dagger, nat
from guppylang.std.quantum import crz, qubit, rz

from guppyalgos.primitives.pauli import pauli_to_z_basis
from guppyalgos.primitives.subroutines.ladders import CXLadderLog, Ladder


def _forward_pauli_exp[n_state_q: nat](
    pauli_string: zqp.String,
    n_qubits: int,
    cx_ladder: type[Ladder] = CXLadderLog,
    rz_method: GuppyFunctionDefinition[[qubit, angle], None] = rz,
) -> GuppyFunctionDefinition[[array[qubit, n_state_q], angle], None]:
    """Build an uncontrolled Trotter term, including measurement-based RZ."""
    if pauli_string.is_identity():
        raise ValueError(
            "Pauli exponential requires at least 1 non-identity Pauli operators"
        )

    basis_change = pauli_to_z_basis(pauli_string=pauli_string, size=n_qubits)

    pauli_indices = list(pauli_string.get_dict().keys())

    @guppy
    @no_type_check
    def pauli_gadget_fn(qreg: array[qubit, comptime(n_qubits)], angle: angle) -> None:
        ladder = cx_ladder()
        basis_change(qreg)
        indices = comptime(pauli_indices)
        # Move the active support out, then restore it before undoing the basis.
        qubit_subset = array(qreg.take(i) for i in indices)

        if len(qubit_subset) > 1:
            ladder.ascending(qubit_subset)

        rz_method(qubit_subset[comptime(len(pauli_indices) - 1)], angle)

        if len(qubit_subset) > 1:
            ladder.ascending_dagger(qubit_subset)

        for i in range(comptime(len(pauli_indices))):
            qreg.put(qubit_subset.take(i), indices[i])
        qubit_subset.discard_all_taken()
        with dagger:
            basis_change(qreg)

    return pauli_gadget_fn


def _controlled_pauli_exp[n_state_q: nat](
    pauli_string: zqp.String,
    n_qubits: int,
    cx_ladder: type[Ladder] = CXLadderLog,
    controlled_rz_method: GuppyFunctionDefinition[[qubit, qubit, angle], None] = crz,
    rz_method: GuppyFunctionDefinition[[qubit, angle], None] = rz,
) -> GuppyFunctionDefinition[[qubit, array[qubit, n_state_q], angle], None]:
    """Build a controlled Trotter term, preserving custom CRZ and identity phases."""
    if pauli_string.is_identity():

        @guppy
        @no_type_check
        def cntrl_pauli_gadget_fn(
            control: qubit,
            qreg: array[qubit, comptime(n_qubits)],
            theta: angle,
        ) -> None:
            rz_method(control, -theta / 2)

        return cntrl_pauli_gadget_fn

    basis_change = pauli_to_z_basis(pauli_string=pauli_string, size=n_qubits)

    pauli_indices = list(pauli_string.get_dict().keys())

    @guppy
    @no_type_check
    def cntrl_pauli_gadget_fn(
        control: qubit,
        qreg: array[qubit, comptime(n_qubits)],
        theta: angle,
    ) -> None:
        ladder = cx_ladder()
        basis_change(qreg)
        indices = comptime(pauli_indices)
        # Move the active support out, then restore it before undoing the basis.
        qubit_subset = array(qreg.take(i) for i in indices)

        if len(qubit_subset) > 1:
            ladder.ascending(qubit_subset)

        controlled_rz_method(
            control, qubit_subset[comptime(len(pauli_indices) - 1)], theta
        )

        if len(qubit_subset) > 1:
            ladder.ascending_dagger(qubit_subset)

        for i in range(comptime(len(pauli_indices))):
            qreg.put(qubit_subset.take(i), indices[i])
        qubit_subset.discard_all_taken()
        with dagger:
            basis_change(qreg)

    return cntrl_pauli_gadget_fn
