"""Register-controlled Givens rotations."""

from __future__ import annotations

from typing import no_type_check

from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, control, dagger, nat
from guppylang.std.quantum import crz, cx, h, qubit, rz, x


@guppy(daggerable=True)
@no_type_check
def _xx_to_z_basis(q0: qubit, q1: qubit) -> None:
    """Map the XX Pauli onto Z on the second qubit."""
    h(q0)
    h(q1)
    cx(q0, q1)


@guppy.unitary
class xx_basis_change:
    """Apply ``exp(-i pi theta XX / 2)`` using a Pauli gadget."""

    @guppy(daggerable=True)
    @no_type_check
    def __call__(theta: angle, q0: qubit, q1: qubit) -> None:
        """Apply the XX rotation."""
        _xx_to_z_basis(q0, q1)
        rz(q1, theta)
        with dagger:
            _xx_to_z_basis(q0, q1)

    @guppy
    @no_type_check
    def controlled[n: nat](
        theta: angle, q0: qubit, q1: qubit, ctrls: array[qubit, n]
    ) -> None:
        """Apply external controls only to the Z rotation."""
        _xx_to_z_basis(q0, q1)
        with control(ctrls):
            rz(q1, theta)
        with dagger:
            _xx_to_z_basis(q0, q1)

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat](
        theta: angle, q0: qubit, q1: qubit, ctrls: array[qubit, n]
    ) -> None:
        """Apply external controls only to the inverse Z rotation."""
        _xx_to_z_basis(q0, q1)
        with control(ctrls), dagger:
            rz(q1, theta)
        with dagger:
            _xx_to_z_basis(q0, q1)


@guppy.struct
class GivensRotationRegisterIncremented[n_data_q: nat]:
    """Build a positive two-target Givens rotation from data-register bits.

    The little-endian data register encodes the integer
    ``x = sum_j 2**j * data_qreg[j]`` and the half-turn parameter
    ``theta = 2 * x / 2**n_data_q``, matching
    :class:`RotationRegisterIncremented`. The rotator diagonalizes the two-qubit
    Givens rotation with an XX basis change, applies opposite-sign
    register-controlled Z rotations, and then restores the original basis. It has
    no private quantum resource and applies no scaling factor.

    """

    @guppy
    @no_type_check
    def compose(
        self,
        data_qreg: array[qubit, n_data_q],
        rotation_regs: tuple[qubit, qubit],
    ) -> None:
        """Apply the register-incremented Givens rotation encoded by ``data_qreg``.

        Args:
            data_qreg: Little-endian register whose bits control the fixed-point
                Givens-angle increments.
            rotation_regs: Pair of target qubits receiving the Givens rotation.

        """
        xx_basis_change(angle(-0.5), rotation_regs[0], rotation_regs[1])

        # Conjugating by X realizes the negative-angle diagonal rotation while
        # retaining the same positive per-bit angle schedule on both targets.
        x(rotation_regs[1])
        for bit_index in range(n_data_q):
            crz(
                data_qreg[bit_index],
                rotation_regs[1],
                angle(2.0 ** (bit_index + 1 - n_data_q)),
            )
        x(rotation_regs[1])

        for bit_index in range(n_data_q):
            crz(
                data_qreg[bit_index],
                rotation_regs[0],
                angle(2.0 ** (bit_index + 1 - n_data_q)),
            )

        xx_basis_change(angle(0.5), rotation_regs[0], rotation_regs[1])

    @guppy.comptime
    @no_type_check
    def daggered(
        self,
        data_qreg: array[qubit, n_data_q],
        rotation_regs: tuple[qubit, qubit],
    ) -> None:
        """Report that the inverse register-incremented rotation is unavailable."""
        raise NotImplementedError
