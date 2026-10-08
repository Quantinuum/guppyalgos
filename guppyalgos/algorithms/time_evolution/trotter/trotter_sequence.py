"""Shared helpers for constructing Trotter product formulas."""

from __future__ import annotations

from typing import no_type_check

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import Function, array, comptime, nat, control
from guppylang.std.quantum import qubit, rz
import zixy.qubit.pauli as zqp

from guppyalgos.primitives.measurement import discard_array_zero
from guppyalgos.primitives.pauli.pauli_exp import pauli_exp
from guppyalgos.primitives.pauli.pauli_exp.pauli_exp import (
    _controlled_pauli_exp_for_function_array,
)
from guppyalgos.primitives.subroutines.ladders import CXLadderLog, Ladder
from guppyalgos.utils import qarray


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
    exponential_terms = [term for term in ham_terms if not term.string.is_identity()]
    exponential_indices: dict[int, int] = {}
    for term_index, term in enumerate(ham_terms):
        if not term.string.is_identity():
            exponential_indices[term_index] = len(exponential_indices)
    exponential_sequence = [
        (exponential_indices[term_index], time_factor)
        for term_index, time_factor in sequence
        if not ham_terms[term_index].string.is_identity()
    ]
    n_exponential_terms = len(exponential_terms)
    n_exponential_steps = len(exponential_sequence)
    identity_phase = sum(
        ham_terms[term_index].coeff * time_factor
        for term_index, time_factor in sequence
        if ham_terms[term_index].string.is_identity()
    )
    rz_flags = getattr(rz_method.wrapped, "unitary_flags", None)
    has_unitary_rz = getattr(rz_flags, "name", None) == "Unitary"

    if n_exponential_steps == 0:
        if not has_unitary_rz:

            @guppy
            @no_type_check
            def empty_trotter_step(
                state_qreg: array[qubit, n_state_qubits], time_step: float
            ) -> None:
                pass

            return empty_trotter_step

        @guppy.unitary
        class phase_only_trotter_step:
            @guppy
            @no_type_check
            def __call__(
                state_qreg: array[qubit, n_state_qubits], time_step: float
            ) -> None:
                pass

            @guppy
            @no_type_check
            def controlled[n_ctrl_q: nat](
                state_qreg: array[qubit, n_state_qubits],
                time_step: float,
                controls: array[qubit, n_ctrl_q],
            ) -> None:
                if n_ctrl_q == 1:
                    rz_method(controls[0], angle(-identity_phase * time_step / 2))
                else:
                    phase_qreg = qarray(1)
                    with control(controls):
                        rz_method(phase_qreg[0], angle(-identity_phase * time_step / 2))
                    discard_array_zero(phase_qreg)

            @guppy
            @no_type_check
            def daggered(
                state_qreg: array[qubit, n_state_qubits], time_step: float
            ) -> None:
                pass

            @guppy
            @no_type_check
            def ctrl_daggered[n_ctrl_q: nat](
                state_qreg: array[qubit, n_state_qubits],
                time_step: float,
                controls: array[qubit, n_ctrl_q],
            ) -> None:
                if n_ctrl_q == 1:
                    rz_method(controls[0], angle(identity_phase * time_step / 2))
                else:
                    phase_qreg = qarray(1)
                    with control(controls):
                        rz_method(phase_qreg[0], angle(identity_phase * time_step / 2))
                    discard_array_zero(phase_qreg)

        return phase_only_trotter_step

    def make_exponential(term: zqp.RealTerm):
        return pauli_exp(term.string, n_state_qubits, cx_ladder, rz_method)

    @guppy.comptime
    @no_type_check
    def pauli_exponentials() -> array[
        Function[[array[qubit, n_state_qubits], angle], None], n_exponential_terms
    ]:
        return [make_exponential(term) for term in exponential_terms]

    if not has_unitary_rz:

        @guppy
        @no_type_check
        def trotter_step(
            state_qreg: array[qubit, n_state_qubits], time_step: float
        ) -> None:
            coeffs = comptime(array(term.coeff for term in exponential_terms))
            term_indices = comptime(
                array(term_index for term_index, _ in exponential_sequence)
            )
            time_factors = comptime(
                array(time_factor for _, time_factor in exponential_sequence)
            )
            exponentials = pauli_exponentials()

            for i in range(n_exponential_steps):
                term_index = term_indices[i]
                exponentials[term_index](
                    state_qreg,
                    angle(coeffs[term_index] * time_factors[i] * time_step),
                )

        return trotter_step

    def make_controlled_exponential(term: zqp.RealTerm, n_ctrl_q: int):
        if n_ctrl_q == 1:
            exponential = _controlled_pauli_exp_for_function_array(
                term.string, n_state_qubits, cx_ladder, rz_method=rz_method
            )

            @guppy
            @no_type_check
            def controlled_exponential(
                state_qreg: array[qubit, n_state_qubits],
                rotation_angle: angle,
                controls: array[qubit, n_ctrl_q],
            ) -> None:
                exponential(controls[0], state_qreg, rotation_angle)

            return controlled_exponential

        exponential = make_exponential(term)

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
        Function[[array[qubit, n_state_qubits], angle, array[qubit, n_ctrl_q]], None],
        n_exponential_terms,
    ]:
        return [
            make_controlled_exponential(term, n_ctrl_q) for term in exponential_terms
        ]

    @guppy.unitary
    class trotter_step:
        @guppy
        @no_type_check
        def __call__(
            state_qreg: array[qubit, n_state_qubits], time_step: float
        ) -> None:
            coeffs = comptime(array(term.coeff for term in exponential_terms))
            term_indices = comptime(
                array(term_index for term_index, _ in exponential_sequence)
            )
            time_factors = comptime(
                array(time_factor for _, time_factor in exponential_sequence)
            )
            exponentials = pauli_exponentials()

            for i in range(n_exponential_steps):
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
            coeffs = comptime(array(term.coeff for term in exponential_terms))
            term_indices = comptime(
                array(term_index for term_index, _ in exponential_sequence)
            )
            time_factors = comptime(
                array(time_factor for _, time_factor in exponential_sequence)
            )
            exponentials = controlled_pauli_exponentials[n_ctrl_q]()

            for i in range(n_exponential_steps):
                term_index = term_indices[i]
                exponentials[term_index](
                    state_qreg,
                    angle(coeffs[term_index] * time_factors[i] * time_step),
                    controls,
                )

            if identity_phase != 0:
                if n_ctrl_q == 1:
                    rz_method(controls[0], angle(-identity_phase * time_step / 2))
                else:
                    phase_qreg = qarray(1)
                    with control(controls):
                        rz_method(phase_qreg[0], angle(-identity_phase * time_step / 2))
                    discard_array_zero(phase_qreg)

        @guppy
        @no_type_check
        def daggered(
            state_qreg: array[qubit, n_state_qubits], time_step: float
        ) -> None:
            coeffs = comptime(array(term.coeff for term in exponential_terms))
            term_indices = comptime(
                array(term_index for term_index, _ in exponential_sequence)
            )
            time_factors = comptime(
                array(time_factor for _, time_factor in exponential_sequence)
            )
            exponentials = pauli_exponentials()

            for i in range(n_exponential_steps):
                term_index = term_indices[i]
                exponentials[term_index](
                    state_qreg,
                    angle(-coeffs[term_index] * time_factors[i] * time_step),
                )

        @guppy
        @no_type_check
        def ctrl_daggered[n_ctrl_q: nat](
            state_qreg: array[qubit, n_state_qubits],
            time_step: float,
            controls: array[qubit, n_ctrl_q],
        ) -> None:
            coeffs = comptime(array(term.coeff for term in exponential_terms))
            term_indices = comptime(
                array(term_index for term_index, _ in exponential_sequence)
            )
            time_factors = comptime(
                array(time_factor for _, time_factor in exponential_sequence)
            )
            exponentials = controlled_pauli_exponentials[n_ctrl_q]()

            for i in range(n_exponential_steps):
                term_index = term_indices[i]
                exponentials[term_index](
                    state_qreg,
                    angle(-coeffs[term_index] * time_factors[i] * time_step),
                    controls,
                )

            if identity_phase != 0:
                if n_ctrl_q == 1:
                    rz_method(controls[0], angle(identity_phase * time_step / 2))
                else:
                    phase_qreg = qarray(1)
                    with control(controls):
                        rz_method(phase_qreg[0], angle(identity_phase * time_step / 2))
                    discard_array_zero(phase_qreg)

    return trotter_step
