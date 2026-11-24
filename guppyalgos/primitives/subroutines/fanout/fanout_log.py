"""Fanout based on log depth cx ladder."""

from guppyalgos.primitives.subroutines.ladders import CXLadderLog
from guppylang.std.builtins import dagger, nat
from typing import no_type_check
from guppylang import guppy, qubit, array
from guppylang.std.quantum import cx


@guppy
@no_type_check
def fanout_log[n_data_qubits: nat](
    control: qubit,
    data_qs: array[qubit, n_data_qubits],
) -> None:
    """Apply log depth fanout to one flat data register."""
    ladder = CXLadderLog()
    with dagger:
        ladder.ascending(data_qs)
    cx(control, data_qs[0])
    ladder.ascending(data_qs)
