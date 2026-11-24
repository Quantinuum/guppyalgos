"""Quantum teleportation implementation."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, nat, owned
from guppylang.std.quantum import cx, h, measure_array, qubit, x, z

from guppyalgos.utils import qarray


@guppy
@no_type_check
def quantum_teleportation[n: nat](
    src: array[qubit, n] @ owned,  # ty: ignore[not-subscriptable]
    out: array[qubit, n],
) -> None:
    r"""Implement quantum teleportation from one register to another.

    Given a source state, transfers its information to the out using an ancillary
    register. It assumes ```out``` is initialized to $\ket{0}$ state. First bell pairs
    are created between ```src``` and ```aux``` and conditional operations are performed
    on ```out``` depending on measurements. Both ```src``` and ```out``` states are
    destroyed.

    Args:
        src (array[qubit, n]): Array of qubits from where to extract the information.
        out (array[qubit, n]): Array of qubits from where to transfer the information.

    """
    aux = qarray(n)

    for i in range(n):
        h(out[i])
        cx(out[i], aux[i])

        cx(src[i], aux[i])
        h(src[i])

    bs_aux = measure_array(aux)
    bs_src = measure_array(src)
    for i in range(n):
        if bs_aux[i]:
            x(out[i])
        if bs_src[i]:
            z(out[i])
