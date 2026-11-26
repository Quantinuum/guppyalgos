"""Shared helpers for LCU and qubitization tests."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import no_type_check

import numpy as np
from numpy.typing import NDArray
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import Function, Unitary, array, comptime, control, nat
from guppylang.std.quantum import qubit, toffoli, x

from guppyalgos.algorithms.select import (
    build_cntrl_select_unary_from_data,
    build_select_unary_from_data,
)
from guppyalgos.algorithms.select.pauli_select import _select_pauli_term


@dataclass(frozen=True)
class LCUData:
    """Classical Hamiltonian data used to build LCU oracles."""

    n_state_qubits: int
    n_terms: int
    n_prep_qubits: int
    pauli_strings: list[zqp.String]
    coeffs: list[float | complex]
    coefficient_phases: list[float]
    l1_norm: float

    @property
    def amplitudes(self) -> NDArray[np.float64]:
        """Return normalized PREPARE amplitudes for the Hamiltonian terms."""
        return np.sqrt(np.abs(self.coeffs) / self.l1_norm)

    @classmethod
    def from_hamiltonian(
        cls,
        ham_op: zqp.RealTermSum | zqp.ComplexTermSum,
    ) -> LCUData:
        """Build classical data for LCU oracles."""
        terms = list(ham_op.to_terms())
        coeffs = [term.coeff for term in terms]

        return cls(
            n_state_qubits=len(ham_op.qubits),
            n_terms=len(terms),
            n_prep_qubits=ceil(log2(len(terms))),
            pauli_strings=[
                term.string  # ty: ignore[unresolved-attribute]
                for term in terms
            ],
            coeffs=coeffs,
            coefficient_phases=[float(np.angle(coeff) / np.pi) for coeff in coeffs],
            l1_norm=sum(abs(coeff) for coeff in coeffs),
        )


def select_with_cntrl_paulis[n_state_q: nat](
    pauli_strings_le: list,
    coefficient_phases: list[float],
    n_state_qubits: int,
    n_terms: int,
) -> GuppyFunctionDefinition[[array[qubit, 1], array[qubit, n_state_q]], None]:
    r"""Build a SELECT oracle for a two-term Hamiltonian using controlled Paulis.

        The single-qubit PREPARE register selects between two Pauli terms. For term
        zero, the PREPARE qubit is flipped so the controlled Pauli acts on
        :math:`|0\rangle`, then flipped back. The Pauli for term one is controlled
        directly on :math:`|1\rangle`. No ancilla qubit is required.

    Args:
        pauli_strings_le: List of zqp.String in little-endian order.
        coefficient_phases: Coefficient phases in half-turns.
        n_state_qubits: Number of qubits in the state register.
        n_terms: Number of Hamiltonian terms.

    Returns:
            GuppyFunctionDefinition for SELECT acting on a single-qubit preparation
            register and the state register.

    """
    controlled_paulis_list = [
        _select_pauli_term(p, n_state_qubits, phase)
        for p, phase in zip(pauli_strings_le, coefficient_phases, strict=True)
    ]

    @guppy.comptime
    @no_type_check
    def _build_cntrl_paulis() -> array[
        Function[[qubit, array[qubit, n_state_qubits]], None], n_terms
    ]:
        return controlled_paulis_list

    @guppy
    @no_type_check
    def select(
        prep: array[qubit, 1],
        state: array[qubit, n_state_qubits],
    ) -> None:
        controlled_paulis = _build_cntrl_paulis()
        x(prep[0])
        controlled_paulis[0](prep[0], state)
        x(prep[0])
        controlled_paulis[1](prep[0], state)

    return select


def select_with_doubly_cntrl_paulis[n_state_q: nat](
    pauli_strings_le: list,
    coefficient_phases: list[float],
    n_state_qubits: int,
    n_terms: int,
) -> GuppyFunctionDefinition[[array[qubit, 2], array[qubit, n_state_q]], None]:
    r"""Build a SELECT oracle using doubly-controlled Paulis.

    The PREPARE register has size :math:`n_{\mathrm{prep}} = 2`. For each
    term :math:`i`, the oracle flips the preparation bits corresponding to
    zero bits of :math:`i`, mapping the selected index to :math:`|11\rangle`.
    It then applies the Pauli using both PREPARE qubits as direct controls
    and restores the PREPARE register. No ancilla qubit is required.

    Args:
        pauli_strings_le: List of zqp.String in little-endian order.
        coefficient_phases: Coefficient phases in half-turns.
        n_state_qubits: Number of qubits in the state register.
        n_terms: Number of Hamiltonian terms.

    Returns:
        GuppyFunctionDefinition for SELECT: (prep: array[qubit,2], state) -> None.

    """

    def add_address_control(
        term: Unitary[[qubit, array[qubit, n_state_q]], None],
        size: int,
    ) -> GuppyFunctionDefinition[[qubit, qubit, array[qubit, n_state_q]], None]:
        @guppy
        @no_type_check
        def apply_term(
            flag: qubit, address: qubit, state: array[qubit, comptime(size)]
        ) -> None:
            with control(flag):
                term(address, state)

        return apply_term

    pauli_terms = [
        add_address_control(
            _select_pauli_term(p, n_state_qubits, phase), n_state_qubits
        )
        for p, phase in zip(pauli_strings_le, coefficient_phases, strict=True)
    ]

    @guppy.comptime
    @no_type_check
    def _build_pauli_terms() -> array[
        Function[[qubit, qubit, array[qubit, n_state_qubits]], None], n_terms
    ]:
        return pauli_terms

    @guppy
    @no_type_check
    def select(prep: array[qubit, 2], state: array[qubit, n_state_qubits]) -> None:
        terms = _build_pauli_terms()
        for i in range(n_terms):
            for bit in range(2):
                if not ((i >> bit) & 1):
                    x(prep[bit])
            terms[i](prep[0], prep[1], state)
            for bit in range(2):
                if not ((i >> bit) & 1):
                    x(prep[bit])

    return select


def build_single_cntrl_select[n_state_q: nat](
    data: LCUData,
) -> GuppyFunctionDefinition[[array[qubit, 1], array[qubit, n_state_q]], None]:
    """Build a single-control SELECT oracle from LCU data."""
    if data.n_terms != 2:
        raise ValueError(f"Expected a two-term Hamiltonian, got {data.n_terms} terms")
    assert data.n_prep_qubits == 1

    return select_with_cntrl_paulis(
        data.pauli_strings,
        data.coefficient_phases,
        data.n_state_qubits,
        data.n_terms,
    )


def build_cntrl_single_cntrl_select[n_state_q: nat](
    data: LCUData,
) -> GuppyFunctionDefinition[[qubit, array[qubit, 1], array[qubit, n_state_q]], None]:
    """Build an externally controlled two-term SELECT oracle from LCU data."""
    if data.n_terms != 2:
        raise ValueError(f"Expected a two-term Hamiltonian, got {data.n_terms} terms")
    assert data.n_prep_qubits == 1
    n_state_qubits = data.n_state_qubits

    first, second = [
        _select_pauli_term(pauli_string, n_state_qubits, phase)
        for pauli_string, phase in zip(
            data.pauli_strings,
            data.coefficient_phases,
            strict=True,
        )
    ]

    @guppy
    @no_type_check
    def cntrl_select(
        flag: qubit,
        prep: array[qubit, 1],
        state: array[qubit, n_state_qubits],
    ) -> None:
        x(prep[0])
        with control(flag):
            first(prep[0], state)
        x(prep[0])
        with control(flag):
            second(prep[0], state)

    return cntrl_select


def build_double_cntrl_select[n_state_q: nat](
    data: LCUData,
) -> GuppyFunctionDefinition[[array[qubit, 2], array[qubit, n_state_q]], None]:
    """Build a doubly-controlled SELECT oracle from LCU data."""
    if not (3 <= data.n_terms <= 4):
        raise ValueError(
            f"Expected a three or four-term Hamiltonian, got {data.n_terms} terms"
        )
    assert data.n_prep_qubits == 2

    return select_with_doubly_cntrl_paulis(
        data.pauli_strings,
        data.coefficient_phases,
        data.n_state_qubits,
        data.n_terms,
    )


def build_unary_iteration_select[n_prep_q: nat, n_state_q: nat](
    data: LCUData,
    comp_and_op: GuppyFunctionDefinition[[qubit, qubit, qubit], None] = toffoli,
    uncomp_and_op: GuppyFunctionDefinition[[qubit, qubit, qubit], None] = toffoli,
) -> GuppyFunctionDefinition[[array[qubit, n_prep_q], array[qubit, n_state_q]], None]:
    """Build a unary-iteration SELECT oracle from LCU data.

    Args:
        data: Classical Hamiltonian data used to build SELECT.
        comp_and_op: Function to compute temporary AND operations.
        uncomp_and_op: Function to uncompute temporary AND operations.

    Returns:
        The unary-iteration SELECT function.

    """
    n_state_qubits = data.n_state_qubits
    n_prep_qubits = data.n_prep_qubits

    if data.n_prep_qubits == 1:
        return select_with_cntrl_paulis(
            data.pauli_strings,
            data.coefficient_phases,
            data.n_state_qubits,
            data.n_terms,
        )

    select_unary = build_select_unary_from_data(
        list(zip(data.pauli_strings, data.coefficient_phases, strict=True)),
        lambda item: _select_pauli_term(item[0], n_state_qubits, item[1]),
        comp_and_op=comp_and_op,
        uncomp_and_op=uncomp_and_op,
    )

    @guppy
    @no_type_check
    def select(
        prep: array[qubit, n_prep_qubits],
        state: array[qubit, n_state_qubits],
    ) -> None:
        select_unary(prep, state)

    return select


def build_cntrl_unary_iteration_select[n_prep_q: nat, n_state_q: nat](
    data: LCUData,
    comp_and_op: GuppyFunctionDefinition[[qubit, qubit, qubit], None] = toffoli,
    uncomp_and_op: GuppyFunctionDefinition[[qubit, qubit, qubit], None] = toffoli,
) -> GuppyFunctionDefinition[
    [qubit, array[qubit, n_prep_q], array[qubit, n_state_q]], None
]:
    """Build an externally controlled unary-iteration SELECT from LCU data.

    Args:
        data: Classical Hamiltonian data used to build SELECT.
        comp_and_op: Function to compute temporary AND operations.
        uncomp_and_op: Function to uncompute temporary AND operations.

    Returns:
        The externally controlled unary-iteration SELECT function.

    """
    n_state_qubits = data.n_state_qubits
    n_prep_qubits = data.n_prep_qubits

    if n_prep_qubits == 1:
        return build_cntrl_single_cntrl_select(data)

    controlled_select_unary = build_cntrl_select_unary_from_data(
        list(zip(data.pauli_strings, data.coefficient_phases, strict=True)),
        lambda item: _select_pauli_term(item[0], n_state_qubits, item[1]),
        comp_and_op=comp_and_op,
        uncomp_and_op=uncomp_and_op,
    )

    @guppy
    @no_type_check
    def cntrl_select(
        control: qubit,
        prep: array[qubit, n_prep_qubits],
        state: array[qubit, n_state_qubits],
    ) -> None:
        controlled_select_unary(control, prep, state)

    return cntrl_select
