"""Nearest-neighbor Givens rotation cascades."""

from __future__ import annotations

from typing import no_type_check

from guppylang.std.builtins import control
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.quantum import cx, qubit

from guppyalgos.primitives.arithmetic.adder.adder_ripple_gidney import (
    adder_ripple_gidney_mod,
)
from guppyalgos.primitives.arithmetic.subtractor.subtractors import (
    cntrl_subtractor_ripple_gidney_mod,
)
from guppyalgos.primitives.rotations.register_incremented_givens_rotation import (
    xx_basis_change,
)
from guppyalgos.utils.guppy.unsafe_borrow import (
    _unsafe_array_borrow,
    _unsafe_array_unborrow,
)


@guppy.comptime
def _assert_valid_cascade_size(
    n_givens: nat @ comptime,
    n_modes: nat @ comptime,
) -> None:
    if n_givens >= n_modes:
        raise ValueError(
            "Givens cascade requires n_givens < n_modes, "
            f"got {n_givens} Givens rotations and {n_modes} modes"
        )


@guppy.struct
class GivensCascadePhaseGradient[
    n_data_q: nat,
    n_givens: nat,
    n_modes: nat,
]:
    """Apply a sequence of positive phase-gradient Givens rotations.

    Each little-endian register ``data_qregs[i]`` encodes an integer ``x_i`` and
    applies a Givens rotation with half-turn parameter
    ``theta_i = 2 * x_i / 2**n_data_q`` to target modes ``i`` and ``i + 1``.
    The rotations are applied in increasing mode order.

    The phase-gradient register is supplied as part of ``rotation_regs``. It is
    borrowed and preserved, allowing one externally prepared resource to be reused
    by this cascade and a later inverse cascade.

    """

    phase_gradient: array[qubit, n_data_q]

    @guppy
    @no_type_check
    def compose(
        self,
        data_qregs: array[array[qubit, n_data_q], n_givens],
        rotation_regs: array[qubit, n_modes],
    ) -> None:
        """Apply the loaded nearest-neighbor Givens cascade.

        Args:
            data_qregs: Little-endian angle registers, one for each neighboring
                target pair in the cascade.
            rotation_regs: Rotation targets and externally prepared phase-gradient
                resource. ``data_qregs[i]`` rotates target modes ``i`` and ``i + 1``.

        """
        _assert_valid_cascade_size(n_givens, n_modes)
        for rotation_index in range(n_givens):
            rotation_targets = (
                rotation_regs.take(rotation_index),
                rotation_regs.take(rotation_index + 1),
            )
            xx_basis_change(
                angle(0.5),
                rotation_targets[1],
                rotation_targets[0],
            )
            cx(rotation_targets[1], rotation_targets[0])

            for bit_index in range(n_data_q):
                cx(
                    rotation_targets[1],
                    self.phase_gradient[bit_index],
                )

            phase_register = _unsafe_array_borrow(self.phase_gradient)
            data_register = data_qregs.take(rotation_index)
            with control(rotation_targets[0]):
                adder_ripple_gidney_mod(data_register, phase_register)
            data_qregs.put(data_register, rotation_index)
            _unsafe_array_unborrow(self.phase_gradient, phase_register)

            for bit_index in range(n_data_q):
                cx(
                    rotation_targets[1],
                    self.phase_gradient[bit_index],
                )

            cx(rotation_targets[1], rotation_targets[0])
            xx_basis_change(
                angle(-0.5),
                rotation_targets[1],
                rotation_targets[0],
            )
            rotation_regs.put(rotation_targets[0], rotation_index)
            rotation_regs.put(rotation_targets[1], rotation_index + 1)

    @guppy
    @no_type_check
    def daggered(
        self,
        data_qregs: array[array[qubit, n_data_q], n_givens],
        rotation_regs: array[qubit, n_modes],
    ) -> None:
        """Undo the loaded nearest-neighbor Givens cascade.

        Args:
            data_qregs: Little-endian angle registers, one for each neighboring
                target pair in the forward cascade.
            rotation_regs: Rotation targets and externally prepared phase-gradient
                resource. ``data_qregs[i]`` inversely rotates target modes ``i`` and
                ``i + 1``.

        """
        _assert_valid_cascade_size(n_givens, n_modes)
        for rotation_index in range(n_givens):
            reverse_index = n_givens - 1 - rotation_index
            rotation_targets = (
                rotation_regs.take(reverse_index),
                rotation_regs.take(reverse_index + 1),
            )
            xx_basis_change(
                angle(0.5),
                rotation_targets[1],
                rotation_targets[0],
            )
            cx(rotation_targets[1], rotation_targets[0])

            for bit_index in range(n_data_q):
                cx(
                    rotation_targets[1],
                    self.phase_gradient[bit_index],
                )

            cntrl_subtractor_ripple_gidney_mod(
                rotation_targets[0],
                data_qregs[reverse_index],
                self.phase_gradient,
            )

            for bit_index in range(n_data_q):
                cx(
                    rotation_targets[1],
                    self.phase_gradient[bit_index],
                )

            cx(rotation_targets[1], rotation_targets[0])
            xx_basis_change(
                angle(-0.5),
                rotation_targets[1],
                rotation_targets[0],
            )
            rotation_regs.put(rotation_targets[0], reverse_index)
            rotation_regs.put(rotation_targets[1], reverse_index + 1)
