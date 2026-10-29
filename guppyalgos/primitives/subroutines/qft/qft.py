"""Quantum Fourier Transform."""

from typing import no_type_check

from guppylang.decorator import guppy
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import qubit, h
from guppylang.std.mem import mem_swap
from guppylang.std.angles import pi
from guppyalgos.primitives.utilities.gates import cphase


@guppy
@no_type_check
def qft[n_qft: nat](qs: array[qubit, n_qft]) -> None:
    """In-place quantum Fourier transform on qs.

    Args:
        qs: Qubit register to perform the QFT on.

    Returns:
        None

    """
    for i in range(n_qft - 1, -1, -1):
        h(qs[i])
        for j in range(i):
            angle = pi / 2 ** (i - j)
            cphase(qs[j], qs[i], angle)
    for i in range(n_qft // 2):
        mem_swap(qs[i], qs[n_qft - i - 1])


@guppy
@no_type_check
def iqft[n_iqft: nat](qs: array[qubit, n_iqft]) -> None:
    """In-place inverse quantum Fourier transform on qs.

    Args:
        qs: Qubit register to perform the IQFT on.

    Returns:
        None

    """
    for k in range(n_iqft // 2):
        mem_swap(qs[k], qs[n_iqft - k - 1])

    for i in range(n_iqft):
        for j in range(i):
            cphase(qs[j], qs[i], -pi / 2 ** (i - j))
        h(qs[i])
