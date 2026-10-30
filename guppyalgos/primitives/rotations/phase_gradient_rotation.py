"""Phase-gradient rotations generated via controlled Gidney addition."""

from __future__ import annotations

from typing import no_type_check

from guppylang.std.builtins import control
from guppylang import guppy
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import qubit, x

from guppyalgos.primitives.arithmetic import (
    adder_ripple_gidney_mod,
    cntrl_subtractor_ripple_gidney_mod,
)
from guppyalgos.utils.guppy.unsafe_borrow import (
    _unsafe_array_borrow,
    _unsafe_array_unborrow,
)

from .rotation_helper import RotationAxis


@guppy.struct
class RotationPhaseGradient[
    n_data_q: nat,
    Axis: RotationAxis,
]:
    r"""Build positive phase-gradient rotations via a controlled Gidney adder.

    The rotator uses the default positive convention: it conjugates the controlled
    adder by ``X`` gates on ``rotation_target`` so the binary value stored in
    ``data_qreg`` is added into the phase-gradient register on the target's original
    ``|0>`` branch. Since the phase-gradient register is a Fourier eigenstate of
    addition, this branch-selective addition kicks back as a positive ``Rz`` phase
    on ``rotation_target`` up to a global phase.

    Writing the standard little-endian integer encoded by ``data_qreg`` as

    .. math:: x = \sum_{j=0}^{d-1} 2^j x_j,

    and the ``d``-qubit phase-gradient register as

    .. math::

        |F_d\rangle = \frac{1}{\sqrt{2^d}}
        \sum_{y=0}^{2^d-1} e^{-2\pi i y / 2^d} |y\rangle,

    modular addition by ``x`` acts diagonally:

    .. math:: A_x |F_d\rangle = e^{-2\pi i x / 2^d} |F_d\rangle.

    With the target-flip convention, the controlled adder maps

    .. math::

        \frac{|0\rangle + |1\rangle}{\sqrt{2}} \otimes |F_d\rangle
        \mapsto
        \frac{e^{-2\pi i x / 2^d}|0\rangle + |1\rangle}{\sqrt{2}}
        \otimes |F_d\rangle,

    which is equivalent, up to global phase, to a positive ``Rz`` on the target
    qubit with half-turn parameter ``theta = 2 * x / 2**d``. The sign is positive
    because the negative addition eigenphase is applied to the original ``|0>``
    branch rather than the original ``|1>`` branch.

    The phase-gradient register must already be prepared in the standard
    little-endian phase-gradient state, for example with
    :func:`guppyalgos.primitives.state_preparation.phase_gradient.phase_gradient` using
    ``Convention.Standard``. Its qubit ordering then matches the arithmetic
    convention of the controlled modular Gidney adder. The kickback circuit
    preserves this resource eigenstate, allowing the same rotator to be applied to
    multiple target qubits. The caller remains responsible for discarding it after
    its final use.

    Args:
        phase_gradient: Externally prepared standard little-endian phase-gradient
            register. The rotator owns and preserves this register.
        axis: Guppy basis-change object that realizes the kicked-back positive
            rotation around the X, Y, or Z axis.

    """

    phase_gradient: array[qubit, n_data_q]
    axis: Axis

    @guppy
    @no_type_check
    def compose(
        self,
        data_qreg: array[qubit, n_data_q],
        rotation_target: qubit,
    ) -> None:
        """Apply the phase-gradient rotation encoded by ``data_qreg``.

        Args:
            data_qreg: Little-endian register encoding the integer rotation value
                ``x``. It has the same width as ``phase_gradient``.
            rotation_target: Qubit receiving the positive axis rotation with
                half-turn parameter ``2 * x / 2**n_data_q``.

        """
        self.axis.prepare_basis(rotation_target)
        x(rotation_target)
        phase_register = _unsafe_array_borrow(self.phase_gradient)
        with control(rotation_target):
            adder_ripple_gidney_mod(data_qreg, phase_register)
        _unsafe_array_unborrow(self.phase_gradient, phase_register)
        x(rotation_target)
        self.axis.restore_basis(rotation_target)

    @guppy
    @no_type_check
    def daggered(
        self,
        data_qreg: array[qubit, n_data_q],
        rotation_target: qubit,
    ) -> None:
        """Apply the phase-gradient rotation encoded by ``data_qreg``.

        Args:
            data_qreg: Little-endian register encoding the integer rotation value
                ``x``. It has the same width as ``phase_gradient``.
            rotation_target: Qubit receiving the positive axis rotation with
                half-turn parameter ``2 * x / 2**n_data_q``.

        """
        self.axis.prepare_basis(rotation_target)
        x(rotation_target)
        cntrl_subtractor_ripple_gidney_mod(
            rotation_target,
            data_qreg,
            self.phase_gradient,
        )
        x(rotation_target)
        self.axis.restore_basis(rotation_target)
