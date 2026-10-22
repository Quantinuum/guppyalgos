"""Temporary logical-AND reduction for adder controls."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, nat, owned
from guppylang.std.quantum import discard, qubit, x

from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_compute,
    temp_and_uncompute,
)


@guppy
@no_type_check
def _acquire_control[m: nat, k: nat](
    controls: array[qubit, m], work: array[qubit, k]
) -> qubit:
    """Borrow one control or compute their conjunction into clean workspace."""
    if m == 1:
        return controls.take(0)
    if m == 0:
        combined = qubit()
        x(combined)
        return combined
    temp_and_compute(controls[0], controls[1], work[0])
    for index in range(m - 2):
        temp_and_compute(work[index], controls[index + 2], work[index + 1])
    return work.take(m - 2)


@guppy
@no_type_check
def _release_control[m: nat, k: nat](
    controls: array[qubit, m],
    work: array[qubit, k],
    combined: qubit @ owned,
) -> None:
    """Restore a borrowed control or measurement-uncompute its conjunction."""
    if m == 1:
        controls.put(combined, 0)
    elif m == 0:
        x(combined)
        discard(combined)
    else:
        work.put(combined, m - 2)
        for index in range(m - 2):
            temp_and_uncompute(
                work[m - 3 - index], controls[m - 1 - index], work[m - 2 - index]
            )
        temp_and_uncompute(controls[0], controls[1], work[0])
