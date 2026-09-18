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
) -> list[float]:
    """Calculate the half-turn rotation angles for eigenvalue inversion.

    Args:
        n_qpe: Number of clock qubits in QPE.
        rotation_scalar: Constant C scaling the inversion factor C / lambda.

    Returns:
        List of 2**n_qpe half-turn angles for Ry rotation on the ancilla qubit.

    """
    dim = 2**n_qpe
    angles = []
    for k in range(dim):
        if k == 0:
            angles.append(0.0)
            continue
        k_signed = k if k < dim // 2 else k - dim
        ratio = rotation_scalar / k_signed
        if abs(ratio) > 1.0:
            raise ValueError(
                f"rotation_scalar is too large for signed clock label {k_signed}"
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
) -> None:
    """Apply the eigenvalue-dependent inversion rotation to an ancilla."""
    angles = comptime(eigenvalue_inversion_angles(n_qpe, rotation_scalar))
    multiplexed_rotation(RotationAxisY(), angles, clock_reg, ancilla)
