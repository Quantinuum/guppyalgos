"""Hamiltonian Simulation using Trotterization."""

from guppylang import guppy
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import qubit
from guppylang.defs import GuppyFunctionDefinition


def ham_sim_trotter[n_state_q: nat](
    trotter_step: GuppyFunctionDefinition[[array[qubit, n_state_q], float], None],
    n_steps: int,
    time_step: float,
    n_state_qubits: int,
) -> GuppyFunctionDefinition[[array[qubit, n_state_q]], None]:
    """Build a full Hamiltonian simulation using Trotter steps.

    This function constructs a Guppy function that simulates the time evolution of a
    quantum system under a given Hamiltonian by repeatedly applying a provided Trotter
    step function. The number of Trotter steps and the time step for each application
    are specified as inputs. The resulting function applies the Trotter step the
    specified number of times to approximate the overall time evolution. Any Trotter
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
        ham_sim = ham_sim_trotter(ham_trotter_step, n_steps, time_step, n_state_qubits)

        @guppy
        @no_type_check
        def main(state_qreg: array[qubit, n_state_qubits]) -> None:
            ham_sim(state_qreg)


    Args:
        trotter_step: A Guppy function implementing a single Trotter step.
        n_steps: The number of Trotter steps to apply.
        time_step: The time step for each Trotter step.
        n_state_qubits: The number of qubits in the quantum state register.

    Returns:
        A Guppy function implementing the full Hamiltonian simulation.

    """

    @guppy
    def ham_sim_fn(state_qreg: array[qubit, n_state_qubits]) -> None:
        for _ in range(n_steps):
            trotter_step(state_qreg, time_step)

    return ham_sim_fn
