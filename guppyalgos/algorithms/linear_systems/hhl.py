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
from guppyalgos.algorithms.linear_systems.hhl_utils import _register_size
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
        controlled_hamiltonian_simulation: Controlled Hamiltonian simulation taking
            a control qubit, the system register, and a signed integer power. Positive
            powers are used for phase estimation and negative powers for uncomputation.
        eigenvalue_inversion: Conditional eigenvalue-inversion rotation taking the
            clock register and a zero-initialized ancilla qubit.

    Returns:
        A Guppy function that applies HHL to an input state register and returns whether
        the algorithm succeeded (ancilla measured 1).

    """
    n_input_qubits = _register_size(controlled_hamiltonian_simulation, 1)
    n_clock_qubits = _register_size(eigenvalue_inversion, 0)

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
