"""Hamiltonian Simulation using Trotterization."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import Function, array, nat
from guppylang.std.quantum import qubit


@guppy
@no_type_check
def ham_sim_trotter[n_state_q: nat](
    state_qreg: array[qubit, n_state_q],
    trotter_step: Function[[array[qubit, n_state_q], float], None],
    n_steps: int,
    time_step: float,
) -> None:
    """Apply a full Hamiltonian simulation using Trotter steps.

    This Guppy function simulates the time evolution of a quantum system under a given
    Hamiltonian by repeatedly applying a provided Trotter step function. Any Trotter
    step function that matches the expected signature can be used, allowing for
    flexibility in the choice of Trotterization method.

    Example:

    .. code-block:: python3

        from guppyalgos.algorithms.time_evolution.trotter import ham_sim_trotter
        from guppyalgos.algorithms.time_evolution.trotter import trotter_first_order
        import zixy.qubit.pauli as zqp

        ham_op = zqp.RealTermSum.from_str(
        "(-0.5, Z0 X1), (-0.1, X0 Z1), (-0.2, Y0 Y1), (-0.3, X0 X1)"
        )
        time_step = 0.1
        n_steps = 3
        n_state_qubits = len(ham_op.qubits)

        ham_trotter_step = trotter_first_order(ham_op, n_state_qubits)

        @guppy
        @no_type_check
        def main(state_qreg: array[qubit, n_state_qubits]) -> None:
            ham_sim_trotter(state_qreg, ham_trotter_step, n_steps, time_step)


    Args:
        state_qreg: The quantum state register to evolve.
        trotter_step: A Guppy function implementing a single Trotter step.
        n_steps: The number of Trotter steps to apply.
        time_step: The time step for each Trotter step.

    Returns:
        None.

    """
    for _ in range(n_steps):
        trotter_step(state_qreg, time_step)
