"""T state preparation."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.quantum import h, qubit, t


@guppy(unitary=True)
@no_type_check
def t_state(qubit: qubit) -> None:
    """Directly generate a T state on a qubit."""
    h(qubit)
    t(qubit)
