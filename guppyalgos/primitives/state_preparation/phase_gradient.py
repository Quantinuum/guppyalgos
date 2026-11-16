"""Phase gradient (Fourier) state preparation."""

from __future__ import annotations

from enum import IntEnum
from typing import no_type_check

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.quantum import h, qubit, rz


class Convention(IntEnum):
    """Register convention for phase-gradient state preparation.

    ``Standard`` is the little-endian phase-gradient layout used by the
    Gidney-adder rotation algorithm to realize the normal positive
    incremented-angle rotation. ``Reciprocal`` uses the reciprocal register
    layout, with the same positive phase schedule assigned directly by qubit
    index.
    """

    Standard = 0
    Reciprocal = 1


def phase_gradient[n_pg: nat](
    n_qubits: int,
    convention: Convention = Convention.Standard,
    rz_method: GuppyFunctionDefinition[[qubit, angle], None] = rz,
) -> GuppyFunctionDefinition[[array[qubit, n_pg]], None]:
    r"""Prepare a phase gradient (Fourier) state on n_qubits qubits.

    The circuit applies a Hadamard gate followed by the positive phase schedule
    ``Rz(π / 2**k)``. With ``Convention.Reciprocal``, this phase is applied to
    qubit ``k`` directly, producing the product state:

    .. math::

        |\mathcal{F}\rangle = \bigotimes_{k=0}^{n-1}
        \frac{|0\rangle + e^{i\pi/2^k}|1\rangle}{\sqrt{2}}

    This is an important resource state for rotation synthesis when used in
    phase-gradient addition circuits.  The Rz synthesis method is invoked
    once per qubit and can be customized via the ``rz_method`` argument,
    matching the interface used in
    :func:`~guppyalgos.primitives.pauli.pauli_exp.pauli_exp`.

    The same positive single-qubit phases are used in both conventions.
    ``Convention.Standard`` is little-endian: it reverses the phase schedule so
    ``qs[0]`` is the least-significant qubit expected by arithmetic-style
    phase-gradient kickback circuits. This is the convention to use with
    :class:`guppyalgos.primitives.rotations.RotationPhaseGradient`
    to realize the normal
    positive incremented-angle rotation. ``Convention.Reciprocal`` assigns the
    positive phase ``π / 2**k`` directly to ``qs[k]``.

    .. code-block:: python3

        from guppyalgos.primitives.state_preparation.phase_gradient import (
            phase_gradient,
        )
        from guppylang.std.quantum import rz

        prep = phase_gradient(4, rz_method=rz)

    Args:
        n_qubits: Number of qubits in the register.
        convention: Register convention used to assign the phase schedule.
            Defaults to :class:`Convention.Standard`.
        rz_method: Rz decomposition method to use for each qubit rotation.
            The callable must have signature ``(qubit, angle) -> None``.
            Defaults to the standard guppy :func:`~guppylang.std.quantum.rz`.

    Returns:
        GuppyFunctionDefinition: A Guppy function that, given an array of
        ``n_qubits`` qubits initialized in the :math:`|0\rangle` state,
        prepares the phase gradient state in-place.

    """

    @guppy.comptime(unitary=True)
    @no_type_check
    def phase_gradient_fn(qs: array[qubit, n_qubits]) -> None:
        for k in range(n_qubits):
            idx = n_qubits - 1 - k if convention == Convention.Standard else k
            h(qs[idx])
            # angle in half-turns: π/2^k rad = 1/2^k half-turns
            rz_method(qs[idx], angle(comptime(1.0 / 2**k)))

    return phase_gradient_fn
