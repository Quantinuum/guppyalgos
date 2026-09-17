"""The HHL algorithm for solving linear systems of equations [1].

References:
    [1] Harrow, A. W., Hassidim, A., & Lloyd, S. (2009). Quantum algorithm for linear
    systems of equations. Physical review letters, 103(15), 150502.

"""

from __future__ import annotations

from typing import no_type_check

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import h, measure, qubit

from guppyalgos.algorithms.phase_estimation import iqpe, qpe
from guppyalgos.algorithms.linear_systems.hhl_utils import register_size
from guppyalgos.primitives.measurement import discard_array_zero
from guppyalgos.utils import qarray, transversal


def hhl[n_input: nat, n_clock: nat](
    controlled_hamiltonian_simulation: GuppyFunctionDefinition[
        [qubit, array[qubit, n_input], int], None
    ],
    eigenvalue_inversion: GuppyFunctionDefinition[[array[qubit, n_clock], qubit], None],
) -> GuppyFunctionDefinition[[array[qubit, n_input]], bool]:
    """Construct a Guppy function for the HHL algorithm.

    Args:
        controlled_hamiltonian_simulation: Guppy function enacting controlled
        Hamiltonian simulation. Explicitly, needs to perform the transformation:
            ``|ctrl⟩|input_state⟩ → |ctrl⟩e^{i * t * ctrl * A}|input_state⟩``
            for some ``t``, where ``A`` is the matrix to be inverted.
        eigenvalue_inversion: Eigenvalue inversion function. Needs to effect:
            ``|λ⟩|0⟩ → |λ⟩(√(1 - |C/λ|^2)|0⟩ + (C/λ)|1⟩)``
            for some scaling factor ``C``, where ``λ`` is the signed eigenvalue
            estimate decoded from the clock register. ``C`` and ``λ`` must use the
            same units, with ``|C/λ| <= 1`` for every supported nonzero ``λ``.


    Returns:
        A Guppy function that applies HHL to an input state register and returns whether
        the algorithm succeeded (ancilla measured 1).

    """
    n_input_qubits = register_size(controlled_hamiltonian_simulation, 1)
    n_clock_qubits = register_size(eigenvalue_inversion, 0)

    @guppy
    @no_type_check
    def inverse_hamiltonian_simulation(
        ctrl: qubit,
        input_state: array[qubit, n_input_qubits],
        power: int,
    ) -> None:
        controlled_hamiltonian_simulation(ctrl, input_state, -power)

    @guppy
    @no_type_check
    def hhl_fn(input_state: array[qubit, n_input_qubits]) -> bool:
        clock_reg = qarray(n_clock_qubits)

        transversal(h, clock_reg)
        qpe(clock_reg, input_state, controlled_hamiltonian_simulation)

        ancilla = qubit()
        eigenvalue_inversion(clock_reg, ancilla)

        iqpe(clock_reg, input_state, inverse_hamiltonian_simulation)
        transversal(h, clock_reg)

        discard_array_zero(clock_reg)
        success = measure(ancilla).read()

        return success

    return hhl_fn
