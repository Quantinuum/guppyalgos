"""Fanout based on log depth cx ladder."""

from typing import no_type_check

from guppylang import array, guppy, qubit
from guppylang.std.builtins import nat
from guppylang.std.quantum import cx

from guppyalgos.primitives.subroutines.ladders import CXLadderLog


@guppy
@no_type_check
def fanout_log[n_data_qubits: nat](
    control: qubit,
    data_qs: array[qubit, n_data_qubits],
) -> None:
    """Apply log depth fanout to one flat data register."""
    ladder = CXLadderLog()
    ladder.ascending_dagger(data_qs)
    cx(control, data_qs[0])
    ladder.ascending(data_qs)
