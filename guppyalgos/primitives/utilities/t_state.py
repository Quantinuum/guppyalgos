"""T state preparation."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.quantum import h, qubit, t


@guppy(unitary=True)
@no_type_check
def t_state(qubit: qubit) -> None:
    """Apply H followed by T; prepares a T state from a zero qubit.

    This circuit is unitary on arbitrary inputs and supports dagger and control.
    """
    h(qubit)
    t(qubit)
