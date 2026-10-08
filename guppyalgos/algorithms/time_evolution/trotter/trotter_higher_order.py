"""Higher-order Suzuki--Trotter product formulas."""

from __future__ import annotations

from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import qubit, rz
import zixy.qubit.pauli as zqp

from guppyalgos.primitives.subroutines.ladders import CXLadderLog, Ladder
from guppyalgos.algorithms.time_evolution.trotter.trotter_sequence import (
    trotter_from_sequence,
)


def scale_sequence(
    sequence: list[tuple[int, float]], factor: float
) -> list[tuple[int, float]]:
    r"""Scale the dimensionless weights stored in a product-formula sequence.

    Each ``(term_index, time_factor)`` entry becomes
    ``(term_index, factor * time_factor)``. This builds a schedule for
    :math:`S(\mathrm{factor} \cdot t)` without changing the runtime
    ``time_step``; the sequence executor applies that time only when it
    calculates each Pauli-exponential angle.

    """
    return [(term_index, factor * time_factor) for term_index, time_factor in sequence]


def suzuki_sequence(n_terms: int, order: int) -> list[tuple[int, float]]:
    """Return Pauli-term indices and time factors for an even-order formula."""
    if order < 2 or order % 2 != 0:
        raise ValueError("Suzuki--Trotter order must be an even integer of at least 2")

    sequence = [(i, 0.5) for i in range(n_terms)]
    sequence.extend((i, 0.5) for i in reversed(range(n_terms)))

    for current_order in range(4, order + 1, 2):
        p_k = 1.0 / (4.0 - 4.0 ** (1.0 / (current_order - 1)))
        outer = scale_sequence(sequence, p_k)
        middle = scale_sequence(sequence, 1.0 - 4.0 * p_k)
        sequence = outer + outer + middle + outer + outer

    return sequence


def trotter_higher_order[n_state_q: nat](
    hamiltonian: zqp.RealTermSum,
    n_state_qubits: int,
    order: int,
    cx_ladder: type[Ladder] = CXLadderLog,
    rz_method: GuppyFunctionDefinition[[qubit, angle], None] = rz,
) -> GuppyFunctionDefinition[[array[qubit, n_state_q], float], None]:
    r"""Build an even-order symmetric Suzuki--Trotter step.

    The second-order formula is the symmetric (Strang) splitting

    .. math::

        S_2(t) = \prod_{j=1}^{m} e^{-i H_j t / 2}
                 \prod_{j=m}^{1} e^{-i H_j t / 2}.

    Higher even orders are generated recursively using

    .. math::

        S_{2k}(t) = S_{2k-2}(p_k t)^2
                    S_{2k-2}((1 - 4p_k)t)
                    S_{2k-2}(p_k t)^2,

    where :math:`p_k = 1 / (4 - 4^{1/(2k-1)})`. The returned Guppy function
    takes the state register and a runtime ``time_step``. Identity terms are
    omitted because they contribute only a global phase to an uncontrolled
    simulation.

    Args:
        hamiltonian: Real Pauli Hamiltonian to simulate.
        n_state_qubits: Number of qubits in the state register.
        order: Desired product-formula order. Must be an even integer at least 2.
        cx_ladder: CX ladder implementation used by each Pauli exponential.
        rz_method: Implementation used for the Pauli-exponential RZ rotations.

    Returns:
        A Guppy function implementing one product-formula step.

    Raises:
        ValueError: If ``order`` is not an even integer of at least 2.

    """
    if not isinstance(order, int) or isinstance(order, bool):
        raise ValueError("Suzuki--Trotter order must be an even integer of at least 2")

    ham_terms: list[zqp.RealTerm] = list(  # ty: ignore[invalid-assignment]
        hamiltonian.to_terms()
    )
    sequence = suzuki_sequence(len(ham_terms), order)
    return trotter_from_sequence(
        ham_terms, sequence, n_state_qubits, cx_ladder, rz_method
    )
