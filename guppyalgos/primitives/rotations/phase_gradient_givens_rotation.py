"""Phase-gradient synthesized Givens rotations."""

from __future__ import annotations

from typing import no_type_check

from guppylang.std.builtins import control
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, nat
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


@guppy.struct
class GivensRotationPhaseGradient[n_data_q: nat]:
    """Build a positive two-target Givens rotation by phase kickback.

    The little-endian data register encodes the integer
    ``x = sum_j 2**j * data_qreg[j]`` and the half-turn parameter
    ``theta = 2 * x / 2**n_data_q``. The rotator diagonalizes the two-qubit Givens
    rotation with an XX basis change, uses controlled Gidney addition to kick back
    the encoded angle, and then restores the original basis.

    ``phase_gradient`` must be prepared externally in the standard
    little-endian phase-gradient state, for example with
    :func:`guppyalgos.primitives.state_preparation.phase_gradient.phase_gradient` using
    ``Convention.Standard``. The kickback circuit preserves it, so one rotator can
    be reused for multiple target pairs. The caller remains responsible for
    discarding the resource register after its final use.

    Args:
        phase_gradient: Externally prepared standard little-endian phase-gradient
            register. The rotator owns and preserves this register.

    """

    phase_gradient: array[qubit, n_data_q]

    @guppy
    @no_type_check
    def compose(
        self,
        data_qreg: array[qubit, n_data_q],
        rotation_regs: tuple[qubit, qubit],
    ) -> None:
        """Apply the phase-gradient Givens rotation encoded by ``data_qreg``.

        Args:
            data_qreg: Little-endian register encoding the integer rotation value
                ``x``. It has the same width as ``phase_gradient``.
            rotation_regs: Pair of target qubits receiving the Givens rotation with
                half-turn parameter ``2 * x / 2**n_data_q``.

        """
        xx_basis_change(angle(0.5), rotation_regs[1], rotation_regs[0])

        cx(rotation_regs[1], rotation_regs[0])

        for bit_index in range(n_data_q):
            cx(rotation_regs[1], self.phase_gradient[bit_index])

        phase_register = _unsafe_array_borrow(self.phase_gradient)
        with control(rotation_regs[0]):
            adder_ripple_gidney_mod(data_qreg, phase_register)
        _unsafe_array_unborrow(self.phase_gradient, phase_register)

        for bit_index in range(n_data_q):
            cx(rotation_regs[1], self.phase_gradient[bit_index])

        cx(rotation_regs[1], rotation_regs[0])

        xx_basis_change(angle(-0.5), rotation_regs[1], rotation_regs[0])

    @guppy
    @no_type_check
    def daggered(
        self,
        data_qreg: array[qubit, n_data_q],
        rotation_regs: tuple[qubit, qubit],
    ) -> None:
        """Undo the phase-gradient Givens rotation encoded by ``data_qreg``.

        Args:
            data_qreg: Little-endian register encoding the integer rotation value
                ``x``. It has the same width as ``phase_gradient``.
            rotation_regs: Pair of target qubits receiving the inverse Givens
                rotation with half-turn parameter ``-2 * x / 2**n_data_q``.

        """
        xx_basis_change(angle(0.5), rotation_regs[1], rotation_regs[0])
        cx(rotation_regs[1], rotation_regs[0])

        for bit_index in range(n_data_q):
            cx(rotation_regs[1], self.phase_gradient[bit_index])

        cntrl_subtractor_ripple_gidney_mod(
            rotation_regs[0],
            data_qreg,
            self.phase_gradient,
        )

        for bit_index in range(n_data_q):
            cx(rotation_regs[1], self.phase_gradient[bit_index])

        cx(rotation_regs[1], rotation_regs[0])
        xx_basis_change(angle(-0.5), rotation_regs[1], rotation_regs[0])
