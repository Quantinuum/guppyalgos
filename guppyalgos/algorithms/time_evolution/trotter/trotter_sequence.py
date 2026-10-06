"""Shared helpers for constructing Trotter product formulas."""

from __future__ import annotations

from typing import no_type_check

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import Function, array, comptime, nat, control
from guppylang.std.quantum import crz, qubit, rz
import zixy.qubit.pauli as zqp

from guppyalgos.primitives.pauli.pauli_exp import cntrl_pauli_exp, pauli_exp
from guppyalgos.primitives.subroutines.ladders import CXLadderLog, Ladder


def trotter_from_sequence[n_state_q: nat](
    ham_terms: list[zqp.RealTerm],
    sequence: list[tuple[int, float]],
    n_state_qubits: int,
    cx_ladder: type[Ladder] = CXLadderLog,
    rz_method: GuppyFunctionDefinition[[qubit, angle], None] = rz,
) -> GuppyFunctionDefinition[[array[qubit, n_state_q], float], None]:
    """Build a Trotter step from Pauli-term indices and time factors.

    The sequence is evaluated from left to right. Each ``(term_index,
    time_factor)`` entry selects a term :math:`c_j P_j` from ``ham_terms`` and
    applies its Pauli-exponential circuit with the runtime angle
    ``c_j * time_factor * time_step``. Repeated indices repeat a term's
    exponential, while negative factors implement a signed backward evolution.

    One Pauli-exponential definition is generated per Hamiltonian term. The
    sequence indices, factors, and Hamiltonian coefficients are embedded into
    the returned Guppy function at compile time; only ``time_step`` is supplied
    when the generated function runs.

    Args:
        ham_terms: Ordered Pauli terms available to the sequence.
        sequence: Ordered ``(term_index, time_factor)`` execution schedule.
        n_state_qubits: Number of qubits in the state register.
        cx_ladder: CX ladder implementation used by each Pauli exponential.
        rz_method: Implementation used for the Pauli-exponential RZ rotations.

    Returns:
        A Guppy function that applies the weighted sequence to a state register.

    """
    n_terms = len(ham_terms)
    n_exponentials = len(sequence)

    if n_terms == 0 or n_exponentials == 0:

        @guppy
        @no_type_check
        def empty_trotter_step(
            state_qreg: array[qubit, n_state_qubits], time_step: float
        ) -> None:
            pass

        return empty_trotter_step

    @guppy.comptime
    @no_type_check
    def pauli_exponentials() -> array[
        Function[[array[qubit, n_state_qubits], angle], None], n_terms
    ]:
        return [
            pauli_exp(term.string, n_state_qubits, cx_ladder, rz_method)
            for term in ham_terms
        ]

    rz_flags = getattr(rz_method.wrapped, "unitary_flags", None)
    if getattr(rz_flags, "name", None) != "Unitary":

        @guppy
        @no_type_check
        def trotter_step(
            state_qreg: array[qubit, n_state_qubits], time_step: float
        ) -> None:
            coeffs = comptime(array(term.coeff for term in ham_terms))
            term_indices = comptime(array(term_index for term_index, _ in sequence))
            time_factors = comptime(array(time_factor for _, time_factor in sequence))
            exponentials = pauli_exponentials()

            for i in range(n_exponentials):
                term_index = term_indices[i]
                exponentials[term_index](
                    state_qreg,
                    angle(coeffs[term_index] * time_factors[i] * time_step),
                )

        return trotter_step

    def make_controlled_exponential(term: zqp.RealTerm, n_ctrl_q: int):
        exponential = pauli_exp(term.string, n_state_qubits, cx_ladder, rz_method)

        @guppy
        @no_type_check
        def controlled_exponential(
            state_qreg: array[qubit, n_state_qubits],
            rotation_angle: angle,
            controls: array[qubit, n_ctrl_q],
        ) -> None:
            with control(controls):
                exponential(state_qreg, rotation_angle)

        return controlled_exponential

    @guppy.comptime
    @no_type_check
    def controlled_pauli_exponentials[n_ctrl_q: nat]() -> array[
        Function[
            [
                array[qubit, n_state_qubits],
                angle,
                array[qubit, n_ctrl_q],
            ],
            None,
        ],
        n_terms,
    ]:
        return [make_controlled_exponential(term, n_ctrl_q) for term in ham_terms]

    @guppy.unitary
    class trotter_step:
        @guppy
        @no_type_check
        def __call__(
            state_qreg: array[qubit, n_state_qubits], time_step: float
        ) -> None:
            coeffs = comptime(array(term.coeff for term in ham_terms))
            term_indices = comptime(array(term_index for term_index, _ in sequence))
            time_factors = comptime(array(time_factor for _, time_factor in sequence))
            exponentials = pauli_exponentials()

            for i in range(n_exponentials):
                term_index = term_indices[i]
                exponentials[term_index](
                    state_qreg,
                    angle(coeffs[term_index] * time_factors[i] * time_step),
                )

        @guppy
        @no_type_check
        def controlled[n_ctrl_q: nat](
            state_qreg: array[qubit, n_state_qubits],
            time_step: float,
            controls: array[qubit, n_ctrl_q],
        ) -> None:
            coeffs = comptime(array(term.coeff for term in ham_terms))
            term_indices = comptime(array(term_index for term_index, _ in sequence))
            time_factors = comptime(array(time_factor for _, time_factor in sequence))
            exponentials = controlled_pauli_exponentials[n_ctrl_q]()

            for i in range(n_exponentials):
                term_index = term_indices[i]
                exponentials[term_index](
                    state_qreg,
                    angle(coeffs[term_index] * time_factors[i] * time_step),
                    controls,
                )

    return trotter_step


def cntrl_trotter_from_sequence[n_state_q: nat](
    ham_terms: list[zqp.RealTerm],
    sequence: list[tuple[int, float]],
    n_state_qubits: int,
    cx_ladder: type[Ladder] = CXLadderLog,
    controlled_rz_method: GuppyFunctionDefinition[[qubit, qubit, angle], None] = crz,
    rz_method: GuppyFunctionDefinition[[qubit, angle], None] = rz,
) -> GuppyFunctionDefinition[[qubit, array[qubit, n_state_q], float], None]:
    """Build a controlled Trotter step from term indices and time factors.

    Sequencing matches :func:`trotter_from_sequence`: entries are executed from
    left to right, and each ``(term_index, time_factor)`` applies the selected
    term with angle ``coefficient * time_factor * time_step``. Repeated and
    negative-weight entries therefore represent repeated and backward
    controlled evolutions, respectively.

    Every scheduled Pauli exponential is conditioned on ``control``. Whether
    identity terms appear in ``ham_terms`` is decided by the caller; retaining
    them preserves their observable phase relative to the inactive control
    branch. The schedule is embedded at compile time and ``time_step`` remains a
    runtime parameter.

    Args:
        ham_terms: Ordered Pauli terms available to the sequence.
        sequence: Ordered ``(term_index, time_factor)`` execution schedule.
        n_state_qubits: Number of qubits in the state register.
        cx_ladder: CX ladder implementation used by each Pauli exponential.
        controlled_rz_method: Implementation used for controlled RZ rotations.
        rz_method: Implementation used for identity-term phases on the control.

    Returns:
        A controlled Guppy function that applies the weighted sequence.

    """
    n_terms = len(ham_terms)
    n_exponentials = len(sequence)

    if n_terms == 0 or n_exponentials == 0:

        @guppy
        @no_type_check
        def empty_cntrl_trotter_step(
            control: qubit,
            state_qreg: array[qubit, n_state_qubits],
            time_step: float,
        ) -> None:
            pass

        return empty_cntrl_trotter_step

    @guppy.comptime
    @no_type_check
    def cntrl_pauli_exponentials() -> array[
        Function[[qubit, array[qubit, n_state_qubits], angle], None], n_terms
    ]:
        return [
            cntrl_pauli_exp(
                term.string,
                n_state_qubits,
                cx_ladder,
                controlled_rz_method,
                rz_method,
            )
            for term in ham_terms
        ]

    @guppy
    @no_type_check
    def cntrl_trotter_step(
        control: qubit,
        state_qreg: array[qubit, n_state_qubits],
        time_step: float,
    ) -> None:
        coeffs = comptime(array(term.coeff for term in ham_terms))
        term_indices = comptime(array(term_index for term_index, _ in sequence))
        time_factors = comptime(array(time_factor for _, time_factor in sequence))
        exponentials = cntrl_pauli_exponentials()

        for i in range(n_exponentials):
            term_index = term_indices[i]
            exponentials[term_index](
                control,
                state_qreg,
                angle(coeffs[term_index] * time_factors[i] * time_step),
            )

    return cntrl_trotter_step
