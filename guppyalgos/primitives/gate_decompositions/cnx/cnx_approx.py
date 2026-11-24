"""Approximate multi-controlled X gate using random XOR method."""

from math import ceil, log2
from typing import no_type_check

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.mem import mem_swap
from guppylang.std.qsystem.random import RNG
from guppylang.std.quantum import cx, qubit, x

from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx


def cnx_approx[n_ctrl: nat](
    n_ctrl: int,
    epsilon: float,
    cnx_method: GuppyFunctionDefinition[[array[qubit, n_ctrl], qubit], None] = cnx,
) -> GuppyFunctionDefinition[[array[qubit, n_ctrl], qubit, RNG], None]:
    r"""Generate a Guppy function to apply an approximate multi-controlled X gate.

    Implements Algorithm 1 from arXiv:2510.07223 "Multi-qubit Toffoli with
    exponentially fewer T gates". The algorithm uses random XOR sampling to
    reduce a large $C^nX$ to a small $C^kX$ where $k = O(\log(1/\varepsilon))$,
    and is independent of $n$. This achieves a $C^nX$ implementation within
    error $\varepsilon$ in the diamond distance using only
    $O(\log(1/\varepsilon))$ T gates instead of $O(n)$.

    Algorithm:
        1. Choose $k = \lceil \log_2(1/\varepsilon) \rceil + 2$ random subsets
           of the control qubits
        2. Compute the XOR parity for each subset (all Clifford operations)
        3. Apply an OR gate to the $k$ parity results using an exact $C^kX$ gate

    Args:
        n_ctrl: Number of control qubits.
        epsilon: Error value for the approximation (diamond distance).
        cnx_method (GuppyFunctionDefinition): Gate implementation to use for the
            exact $C^kX$ gate.

    Returns:
        GuppyFunctionDefinition[[array[qubit, n_ctrl], qubit, RNG], None]:
            Function taking an array of control qubits, a target qubit, and a
            random number generator; applies approximate $C^nX$ gate.

    """
    if not 0 < epsilon < 1:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
    if n_ctrl < 1:
        raise ValueError(f"n_ctrl must be >= 1, got {n_ctrl}")

    k = min(ceil(log2(1 / epsilon)) + 2, n_ctrl)

    @guppy.comptime
    @no_type_check
    def _cnx_on_first_k(
        controls: array[qubit, n_ctrl],
        target: qubit,
    ) -> None:
        cnx_method([controls[i] for i in range(k)], target)

    @guppy
    @no_type_check
    def cnx_approx_fn(
        controls: array[qubit, n_ctrl],
        target: qubit,
        rng: RNG,
    ) -> None:
        # If k == n_ctrl, we use an exact CnX gate
        if comptime(k == n_ctrl):
            cnx_method(controls, target)
            return

        # Flip all controls
        for i in range(n_ctrl):
            x(controls[i])

        # Randomly permute controls to select the first k qubits randomly
        indices = array(i for i in range(n_ctrl))
        rng.shuffle(indices)
        for i in range(k):
            if indices[i] != i:
                mem_swap(controls[i], controls[indices[i]])

        # Apply random CX gates to create random parities on the first k qubits
        cx_applied = array((-1, -1) for _ in range(comptime(k * (n_ctrl - 1))))
        cx_count = 0
        for targ in range(k):
            for ctrl in range(n_ctrl):
                if ctrl != targ and rng.random_int_bounded(2) == 1:
                    cx(controls[ctrl], controls[targ])
                    cx_applied[cx_count] = (ctrl, targ)
                    cx_count = cx_count + 1

        # Flip first k controls
        for i in range(k):
            x(controls[i])

        # Apply exact CkX gate
        _cnx_on_first_k(controls, target)

        # Unflip first k controls
        for i in range(k):
            x(controls[i])

        # Uncompute random CX gates in reverse order
        for i in range(cx_count):
            ctrl, targ = cx_applied[cx_count - 1 - i]
            cx(controls[ctrl], controls[targ])

        # Undo the random permutation
        for i in range(k - 1, -1, -1):
            if indices[i] != i:
                mem_swap(controls[i], controls[indices[i]])

        # Flip all controls back
        for i in range(n_ctrl):
            x(controls[i])

    return cnx_approx_fn
