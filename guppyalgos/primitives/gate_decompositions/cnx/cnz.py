"""Multicontrolled z gate."""

from typing import no_type_check

from guppylang.decorator import guppy
from guppylang.std.builtins import Function, array, nat
from guppylang.std.quantum import h, qubit


@guppy
@no_type_check
def cnz[n_controls: nat](
    controls: array[qubit, n_controls],
    target: qubit,
    cnx_box: Function[[array[qubit, n_controls], qubit], None],
) -> None:
    r"""Apply cnz by conjugating the provided cnx_box.

    Args:
        controls: The array of control qubits.
        target: The target qubit.
        cnx_box: The cnx box to use for the decomposition.

    """
    h(target)
    cnx_box(controls, target)
    h(target)
