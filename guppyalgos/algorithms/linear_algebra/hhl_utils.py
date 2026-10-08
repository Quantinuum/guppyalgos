"""Guppy function builders used to configure HHL."""

from __future__ import annotations

from typing import no_type_check

import numpy as np
from guppylang import guppy
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.quantum import qubit

from guppyalgos.primitives.rotations import RotationAxisY, multiplexed_rotation


def eigenvalue_inversion_angles(
    n_qpe: int,
    rotation_scalar: float,
    simulation_time: float,
) -> list[float]:
    """Calculate the half-turn rotation angles for eigenvalue inversion.

    Args:
        n_qpe: Number of clock qubits in QPE.
        rotation_scalar: Constant C scaling the inversion factor C / lambda.
        simulation_time: Time t used by the Hamiltonian simulation in QPE.

    Returns:
        List of 2**n_qpe half-turn angles for Ry rotation on the ancilla qubit.

    """
    dim = 2**n_qpe
    angles = []
    for k in range(dim):
        if k == 0:
            angles.append(0.0)
            continue
        signed_label = k if k < dim // 2 else k - dim
        eigenvalue = 2.0 * np.pi * signed_label / (simulation_time * dim)
        ratio = rotation_scalar / eigenvalue
        if abs(ratio) > 1.0:
            raise ValueError(
                f"rotation_scalar is too large for signed clock label {signed_label}"
            )
        half_turn = (2.0 / np.pi) * np.arcsin(ratio)
        angles.append(float(half_turn))
    return angles


@guppy
@no_type_check
def eigenvalue_inversion[n_qpe: nat](
    clock_reg: array[qubit, n_qpe],
    ancilla: qubit,
    rotation_scalar: float @ comptime,
    simulation_time: float @ comptime,
) -> None:
    """Rotate an ancilla by the reciprocal eigenvalue encoded in ``clock_reg``.

    For a little-endian clock basis state ``|b>`` of ``m = n_qpe`` qubits, its
    unsigned value ``k`` is interpreted as the signed integer ``b = k`` when
    ``k < 2**(m - 1)`` and ``b = k - 2**m`` otherwise. The encoded eigenvalue is
    ``lambda = 2 pi b / (simulation_time * 2**m)``. This function performs
    ``|b>|0> -> |b>(sqrt(1 - (C/lambda)**2)|0> + C/lambda|1>)``, where
    ``C = rotation_scalar``. The zero label leaves the ancilla unchanged.

    Args:
        clock_reg: Little-endian register containing the signed phase estimate.
        ancilla: Target qubit initialized to zero.
        rotation_scalar: Constant C in the reciprocal-eigenvalue amplitude.
        simulation_time: Hamiltonian simulation time t used by QPE.

    """
    angles = comptime(
        eigenvalue_inversion_angles(n_qpe, rotation_scalar, simulation_time)
    )
    multiplexed_rotation(RotationAxisY(), angles, clock_reg, ancilla)
