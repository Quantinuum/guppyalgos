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
from guppylang.std.angles import angle
from guppylang.std.builtins import Function, array, nat
from guppylang.std.quantum import h, qubit, s, sdg, toffoli, x

from guppyalgos.utils import cphase
from guppyalgos.algorithms.select import (
    build_cntrl_select_unary_from_data,
    build_select_unary_from_data,
)
from guppyalgos.primitives.pauli import pauli_to_cntrl_gate


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
        pauli_to_cntrl_gate(p, n_state_qubits, phase)
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


def get_doubly_cntrl_paulis[n_state_q: nat](
    pauli_string: zqp.String,
    size: int,
    phase: float = 0.0,
) -> GuppyFunctionDefinition[[qubit, qubit, array[qubit, n_state_q]], None]:
    r"""Build a gate that applies :math:`P` when both controls are on.

    For each nonidentity Pauli acting on target :math:`q_j`, the
    doubly-controlled operation is decomposed as

    .. math::

        \begin{aligned}
        \operatorname{CCX}_{c_0,c_1,q_j} &=
            \operatorname{Toffoli}(c_0,c_1,q_j), \\
        \operatorname{CCZ}_{c_0,c_1,q_j} &=
            H_{q_j}\operatorname{Toffoli}(c_0,c_1,q_j)H_{q_j}, \\
        \operatorname{CCY}_{c_0,c_1,q_j} &=
            S_{q_j}\operatorname{Toffoli}(c_0,c_1,q_j)S^\dagger_{q_j}.
        \end{aligned}

    Args:
        pauli_string: zixy Pauli string.
        size: Number of qubits in the state register.
        phase: Coefficient phase in half-turns.

    Returns:
        GuppyFunctionDefinition for (ctrl0, ctrl1, state) -> None.

    """

    @guppy.comptime
    @no_type_check
    def doubly_cntrl_paulis(ctrl0: qubit, ctrl1: qubit, q: array[qubit, size]) -> None:
        for idx, p in pauli_string.get_dict().items():
            match p:
                case zqp.X:
                    toffoli(ctrl0, ctrl1, q[idx])
                case zqp.Z:
                    h(q[idx])
                    toffoli(ctrl0, ctrl1, q[idx])
                    h(q[idx])
                case zqp.Y:
                    # CCY = S · CCX · S† on target
                    sdg(q[idx])
                    toffoli(ctrl0, ctrl1, q[idx])
                    s(q[idx])
        cphase(ctrl0, ctrl1, angle(phase))

    return doubly_cntrl_paulis


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
    cc_gates_list = [
        get_doubly_cntrl_paulis(p, n_state_qubits, phase)
        for p, phase in zip(pauli_strings_le, coefficient_phases, strict=True)
    ]

    @guppy.comptime
    @no_type_check
    def _build_cc_paulis() -> array[
        Function[[qubit, qubit, array[qubit, n_state_qubits]], None], n_terms
    ]:
        return cc_gates_list

    @guppy
    @no_type_check
    def select(prep: array[qubit, 2], state: array[qubit, n_state_qubits]) -> None:
        cc_paulis = _build_cc_paulis()
        for i in range(n_terms):
            for bit in range(2):
                if not ((i >> bit) & 1):
                    x(prep[bit])
            cc_paulis[i](prep[0], prep[1], state)
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

    controlled_paulis = [
        get_doubly_cntrl_paulis(pauli_string, n_state_qubits, phase)
        for pauli_string, phase in zip(
            data.pauli_strings,
            data.coefficient_phases,
            strict=True,
        )
    ]

    @guppy.comptime
    @no_type_check
    def cntrl_pauli_gates() -> array[
        Function[[qubit, qubit, array[qubit, n_state_qubits]], None], 2
    ]:
        return controlled_paulis

    @guppy
    @no_type_check
    def cntrl_select(
        control: qubit,
        prep: array[qubit, 1],
        state: array[qubit, n_state_qubits],
    ) -> None:
        gates = cntrl_pauli_gates()
        x(prep[0])
        gates[0](control, prep[0], state)
        x(prep[0])
        gates[1](control, prep[0], state)

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
    and_op: GuppyFunctionDefinition[[qubit, qubit, qubit], None] = toffoli,
) -> GuppyFunctionDefinition[[array[qubit, n_prep_q], array[qubit, n_state_q]], None]:
    """Build a unary-iteration SELECT oracle from LCU data.

    Args:
        data: Classical Hamiltonian data used to build SELECT.
        and_op: AND unitary whose dagger uncomputes each work qubit.

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
        lambda item: pauli_to_cntrl_gate(item[0], n_state_qubits, item[1]),
        and_op=and_op,
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
    and_op: GuppyFunctionDefinition[[qubit, qubit, qubit], None] = toffoli,
) -> GuppyFunctionDefinition[
    [qubit, array[qubit, n_prep_q], array[qubit, n_state_q]], None
]:
    """Build an externally controlled unary-iteration SELECT from LCU data.

    Args:
        data: Classical Hamiltonian data used to build SELECT.
        and_op: AND unitary whose dagger uncomputes each work qubit.

    Returns:
        The externally controlled unary-iteration SELECT function.

    """
    n_state_qubits = data.n_state_qubits
    n_prep_qubits = data.n_prep_qubits

    if n_prep_qubits == 1:
        return build_cntrl_single_cntrl_select(data)

    controlled_select_unary = build_cntrl_select_unary_from_data(
        list(zip(data.pauli_strings, data.coefficient_phases, strict=True)),
        lambda item: pauli_to_cntrl_gate(item[0], n_state_qubits, item[1]),
        and_op=and_op,
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
