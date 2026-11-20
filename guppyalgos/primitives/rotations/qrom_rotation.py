"""Generic QROM compute -> rotate -> uncompute helper."""

from __future__ import annotations
from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import Function


@guppy(unitary=True)
@no_type_check
def qrom_identity[IndexRegs, TargetRegs](
    index_regs: IndexRegs,
    target_regs: TargetRegs,
) -> None:
    """Leave QROM index and target registers unchanged."""


@guppy.protocol
class Rotator[TargetRegs, RotationRegs]:
    """Operation that turns a loaded QROM word into a target rotation."""

    @guppy.require
    @no_type_check
    def compose(
        self,
        data_qreg: TargetRegs,
        rotation_regs: RotationRegs,
    ) -> None:
        """Apply a rotation encoded in ``data_qreg`` to ``rotation_regs``."""
        ...

    @guppy.require
    @no_type_check
    def daggered(
        self,
        data_qreg: TargetRegs,
        rotation_regs: RotationRegs,
    ) -> None:
        """Apply dagger of a rotation encoded in ``data_qreg`` to ``rotation_regs``."""
        ...


@guppy.struct
class QROMRotations[
    IndexRegs,
    TargetRegs,
    RotationRegs,
    RotBox: Rotator[TargetRegs, RotationRegs],  # ty: ignore[invalid-type-variable-bound]
]:
    """Composable QROM rotation gadget.

    A QROM rotation takes a set of rotation parameters indexed on an index register,
    stored in one or more QROM target registers, applies a rotation controlled on
    those rotation increments in superposition, and then uncomputes the target
    registers.

    The struct factors an indexed rotation into three pieces:

    1. a QROM compute callable that maps ``(index_qreg, data_qreg)`` to the
       selected target data,
    2. a rotator object that applies the loaded word to its target registers, and
    3. a matching QROM uncompute callable that restores ``data_qreg`` after the
       rotation has been applied.

    ``IndexRegs`` describes the QROM index-register shape, while ``TargetRegs``
    describes the QROM-loaded register shape. Future QROM implementations may use
    structs containing multiple index or target registers. Current implementations
    bind each to one ``array[qubit, n]``. ``RotationRegs`` similarly describes the
    register shape receiving the rotation; current implementations use a single
    ``qubit`` for single-qubit rotations or ``tuple[qubit, qubit]`` for Givens
    rotation variants.

    Supported rotation cases:

    Register-incremented rotators have no stored quantum resource.
    Phase-gradient rotators instead own an externally prepared phase-gradient
    register, which remains reusable across calls to :meth:`compose`.

    Args:
        qrom_compute: Callable that loads the indexed rotation data into the QROM
            target registers.
        rotation_box: Rotator struct that converts the loaded data into the requested
            target rotation.
        qrom_uncompute: Callable that restores the QROM target registers after the
            rotation.

    """

    qrom_compute: Function[[IndexRegs, TargetRegs], None]
    rotation_box: RotBox
    qrom_uncompute: Function[[IndexRegs, TargetRegs], None]

    @guppy
    @no_type_check
    def compose(
        self,
        index_qreg: IndexRegs,
        data_qreg: TargetRegs,
        rotation_regs: RotationRegs,
    ) -> None:
        """Apply QROM compute, rotation, and QROM uncompute.

        Args:
            index_qreg: The QROM index register or register bundle.
            data_qreg: The QROM target register or register bundle.
            rotation_regs: Target register shape expected by the chosen rotator.

        """
        self.qrom_compute(index_qreg, data_qreg)
        self.rotation_box.compose(data_qreg, rotation_regs)
        self.qrom_uncompute(index_qreg, data_qreg)

    @guppy
    @no_type_check
    def daggered(
        self,
        index_qreg: IndexRegs,
        data_qreg: TargetRegs,
        rotation_regs: RotationRegs,
    ) -> None:
        """Apply the inverse QROM rotation in reverse operation order.

        Args:
            index_qreg: The QROM index register or register bundle.
            data_qreg: The QROM target register or register bundle.
            rotation_regs: Target register shape expected by the chosen rotator.

        """
        self.qrom_uncompute(index_qreg, data_qreg)
        self.rotation_box.daggered(data_qreg, rotation_regs)
        self.qrom_compute(index_qreg, data_qreg)
