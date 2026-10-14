"""First-order Trotterization for Hamiltonian Simulation."""

from __future__ import annotations

from guppylang.std.angles import angle
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import qubit, rz
from guppylang.defs import GuppyFunctionDefinition
import zixy.qubit.pauli as zqp

from guppyalgos.primitives.subroutines.ladders import CXLadderLog, Ladder
from guppyalgos.algorithms.time_evolution.trotter.trotter_sequence import (
    trotter_from_sequence,
)


def trotter_first_order[n_state_q: nat](
    hamiltonian: zqp.RealTermSum,
    n_state_qubits: int | None = None,
    cx_ladder: type[Ladder] = CXLadderLog,
    rz_method: GuppyFunctionDefinition[[qubit, angle], None] = rz,
) -> GuppyFunctionDefinition[[array[qubit, n_state_q], float], None]:
    """Build a single first-order Trotter step for Hamiltonian simulation.

    This function constructs a Trotter step function that can be used to simulate the
    time  evolution of a quantum system under a given Real Hamiltonian using first-order
    Trotterization. It is constructed by exponentiating each term in the Hamiltonian
    sequentially using the `pauli_exp` function to create the exponentiation circuits
    for each Pauli term. The defaults for `cx_ladder_method` and `rz_method` are set to
    use logarithmic CX ladders and standard RZ gates, respectively.

    Identity terms do not affect the uncontrolled state evolution. When this unitary
    is controlled through its custom modifier, identity terms contribute the
    corresponding phase relative to the inactive control branch.

    Example:

    .. code-block:: python3

        from guppyalgos.algorithms.time_evolution.trotter import trotter_first_order
        import zixy.qubit.pauli as zqp
        n_state_qubits = 4
        hamiltonian = zqp.RealTermSum.from_str("(-0.5, Z0 X1), (-0.1, X0 Z1)")
        trotter_step = trotter_first_order(hamiltonian, n_state_qubits)

    Args:
        hamiltonian (zqp.RealTermSum): The Hamiltonian to simulate, represented as a
        sum of Pauli operators.
        n_state_qubits (int): The number of qubits in the quantum state register.
        cx_ladder (Ladder): CX ladder implementing Ladder protocol.
        rz_method (Callable, optional): Method to implement RZ rotations. Default rz.

    Returns:
        A Guppy function implementing the Trotter step.

    """
    n_qubits = len(hamiltonian.qubits) if n_state_qubits is None else n_state_qubits

    ham_terms: list[zqp.RealTerm] = list(hamiltonian.to_terms())  # ty: ignore[invalid-assignment]

    sequence = [(i, 1.0) for i in range(len(ham_terms))]
    return trotter_from_sequence(ham_terms, sequence, n_qubits, cx_ladder, rz_method)
