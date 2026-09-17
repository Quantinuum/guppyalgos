"""GHZ state preparation using a CX ladder implementation."""

from typing import no_type_check

from guppylang import comptime, guppy
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import h, qubit

from guppyalgos.primitives.subroutines.ladders.cx_ladder import (
    _cx_ladder_apply_from_inds,
    _ghz_state_prep_filter,
    log_cx_ladder_indices,
)


@guppy(unitary=True)
@no_type_check
def ghz_state[n: nat](
    q: array[qubit, n],
) -> None:
    r"""Build a GHZ state from $\ket{0}$ using a log depth CX ladder.

    Contains a filter to remove unnecessary gates from the usual log depth CX ladder
    as the input state is all 0, this reduces the gate count to the same as a linear
    ladder, while retaining log depth.

    GHZ state on n-qubits is given by:
    $\ket{GHZ} = (\ket{0 ... 0} + \ket{1 ... 1}) / √2$

    Args:
        q (array[qubit, n]): Array of qubits to prepare the GHZ state on.

    Due to https://github.com/Quantinuum/guppylang/issues/2225, needs to be called on
    arrays of size > 1.

    """
    filtered_cx_gate_indices = comptime(
        _ghz_state_prep_filter(log_cx_ladder_indices(n))
    )
    h(q[0])
    _cx_ladder_apply_from_inds(q, filtered_cx_gate_indices)
