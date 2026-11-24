"""Dicke State Preparation."""

from typing import no_type_check

import numpy as np
from guppylang import guppy
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.quantum import angle, crz, cx, qubit, rx, x

from guppyalgos.primitives.gate_decompositions.cnx.ccu import ccry_toffoli


@guppy
@no_type_check
def dicke_nk[n: nat](qs: array[qubit, n], k: int @ comptime) -> None:
    """Prepare a Dicke state on a given register for k excitations.

    The implementation is based on https://arxiv.org/pdf/1904.07358v1.
    However, notice our construction is top to bottom instead of bottom
    to top, so the k qubits flipped are the first ones and not the last
    ones. Dicke states are symmetrical so the outcome is the same.

    It relies on two building blocks, the unitaries U_nk and the Split
    and Cyclic Shift gates.

    Args:
        qs (array of qubits): Register of qubits where to prepare the state.
        k (int): Number of excitations.

    """
    for i in range(k):
        x(qs[i])
    u_nk(qs, k)


@guppy.comptime
@no_type_check
def u_nk[n: nat](qs: array[qubit, n], k: int @ comptime) -> None:
    """Apply the unitary U_nk."""
    for j in range(n, k, -1):
        scs_m_t(qs, j, k)
    for j in range(k, 1, -1):
        scs_m_t(qs, j, j - 1)


@guppy.comptime
@no_type_check
def scs_m_t[n: nat](qs: array[qubit, n], m: int @ comptime, t: int @ comptime) -> None:
    """Apply a Split and Cyclic Shift."""
    cx(qs[n - m + 1], qs[n - m])

    # cry
    theta = angle(2 * np.arccos(np.sqrt(1 / m)) / np.pi)
    rx(qs[n - m + 1], angle(1 / 2))
    crz(qs[n - m], qs[n - m + 1], theta)
    rx(qs[n - m + 1], angle(-1 / 2))

    cx(qs[n - m + 1], qs[n - m])

    for ll in range(2, t + 1):
        theta = angle(2 * np.arccos(np.sqrt(ll / m)) / np.pi)
        cx(qs[n - m + ll], qs[n - m])
        ccry_toffoli(qs[n - m], qs[n - m + ll - 1], qs[n - m + ll], theta)
        cx(qs[n - m + ll], qs[n - m])
