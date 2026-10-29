"""Reflection box based on the CnZ implementation."""

from __future__ import annotations

from typing import no_type_check

from guppylang.decorator import guppy
from guppylang.std.builtins import Function, array, comptime, nat
from guppylang.std.quantum import qubit, x

from guppyalgos.primitives.gate_decompositions.cnx.cnz import cnz
from guppyalgos.utils import transversal
from guppyalgos.utils.guppy.unsafe_borrow import (
    _unsafe_array_borrow,
    _unsafe_array_unborrow,
)


@guppy
@no_type_check
def reflect_zero[n_controls: nat](
    controls: array[qubit, n_controls],
    target: qubit,
    cnx_box: Function[[array[qubit, n_controls], qubit], None],
) -> None:
    """Apply I - 2|0...0><0...0| to separate controls and target.

    Flip the sign only when all controls and the target are zero. With no
    controls, this applies XZX = -Z to the target. The supplied controlled-X
    implementation determines the decomposition and any ancilla requirements.
    """
    transversal(x, controls)
    x(target)
    cnz(controls, target, cnx_box)
    x(target)
    transversal(x, controls)


@guppy
@no_type_check
def reflection_box[n_qubits: nat](
    qreg: array[qubit, n_qubits],
    cnx_box: Function[[array[qubit, comptime(n_qubits - 1)], qubit], None],  # ty: ignore[unsupported-operator]
) -> None:
    r"""Apply a Householder reflection about the all-zero state.

    The reflection operator is

    .. math::

        R = I - 2\lvert 0^n\rangle\langle 0^n\rvert,

    where :math:`\lvert 0^n\rangle` is the all-zero state and :math:`I` is the
    identity operator. It flips the sign of the all-zero state and leaves all
    other states unchanged. For one qubit, :math:`R = -Z`.

    The reflection is implemented using a multi-controlled :math:`Z` gate with
    open controls, decomposed using ``cnx_box``. Thus, the gate is applied when
    every control qubit is in state :math:`\lvert 0\rangle`.

    """
    borrowed_qreg = _unsafe_array_borrow(qreg)
    controls = array(borrowed_qreg.take(i) for i in range(comptime(n_qubits - 1)))
    target = borrowed_qreg.take(n_qubits - 1)

    reflect_zero(controls, target, cnx_box)

    i = 0
    for q in controls:
        borrowed_qreg.put(q, i)
        i += 1
    borrowed_qreg.put(target, n_qubits - 1)

    _unsafe_array_unborrow(qreg, borrowed_qreg)


@guppy.struct
@no_type_check
class Reflection[n_qubits: nat]:
    r"""Guppy struct representing a reflection operator.

    The operator is

    .. math::

        R = I - 2\lvert 0^n\rangle\langle 0^n\rvert

    and is used in the qubitization walk operator :math:`W = R L`.
    """

    cnx_method: Function[[array[qubit, comptime(n_qubits - 1)], qubit], None]  # ty: ignore[unsupported-operator]

    @guppy
    @no_type_check
    def compose(
        self,
        qreg: array[qubit, n_qubits],
    ) -> None:
        r"""Apply the reflection operator :math:`R`.

        Args:
            qreg: The qubit register on which to apply the reflection.

        """
        reflection_box(qreg, self.cnx_method)


@guppy
@no_type_check
def cntrl_reflection_box[n_qubits: nat](
    control: qubit,
    qreg: array[qubit, n_qubits],
    cnx_box: Function[[array[qubit, n_qubits], qubit], None],
) -> None:
    r"""Apply an externally controlled reflection about the all-zero state.

    This implements the joint operator

    .. math::

        C(R) = \lvert 0\rangle\!\langle 0\rvert \otimes I
            + \lvert 1\rangle\!\langle 1\rvert \otimes
            \left(I - 2\lvert 0^n\rangle\!\langle 0^n\rvert\right),

    where the first subsystem is ``control`` and the second is ``qreg``.
    Equivalently, the amplitude of
    :math:`\lvert 1\rangle\lvert 0^n\rangle` is negated and every other
    computational-basis amplitude is unchanged. If ``control`` is in a
    superposition, this conditional sign becomes a relative phase and may
    entangle it with ``qreg``.

    The qubits in ``qreg`` are implemented as open controls by conjugating
    them with :math:`X`. The resulting multi-controlled :math:`Z` uses
    ``qreg`` as its controls, ``control`` as its target, and ``cnx_box`` for
    the underlying multi-controlled :math:`X` decomposition. All arguments
    are borrowed and restored in place; the function allocates no persistent
    output register.

    Args:
        control: External control, active on :math:`\lvert 1\rangle`.
        qreg: Register reflected about :math:`\lvert 0^n\rangle` when the
            external control is active.
        cnx_box: Decomposition of an ``n_qubits``-controlled :math:`X`, used
            to implement the multi-controlled :math:`Z`.

    """
    transversal(x, qreg)
    cnz(qreg, control, cnx_box)
    transversal(x, qreg)


@guppy.struct
@no_type_check
class ReflectionCntrl[n_qubits: nat]:
    r"""Externally controlled reflection about an all-zero register state.

    The struct packages the decomposition needed to apply
    :func:`cntrl_reflection_box` to an arbitrary ``n_qubits`` register. Its
    ``compose`` method negates only the joint basis state
    :math:`\lvert 1\rangle\lvert 0^n\rangle`, where the first qubit is the
    external control. It therefore acts as the identity when that control is
    :math:`\lvert 0\rangle` and as
    :math:`I - 2\lvert 0^n\rangle\!\langle 0^n\rvert` when it is
    :math:`\lvert 1\rangle`.

    This callable wrapper is used by controlled qubitization to apply the
    reflection part of the walk operator conditionally. Both the external
    control and reflected register are borrowed and mutated in place.

    Attributes:
        cnx_method: Decomposition of an ``n_qubits``-controlled :math:`X`
            used by the controlled reflection.

    """

    cnx_method: Function[[array[qubit, n_qubits], qubit], None]

    @guppy
    @no_type_check
    def compose(
        self,
        control: qubit,
        qreg: array[qubit, n_qubits],
    ) -> None:
        r"""Apply the controlled all-zero reflection in place.

        Args:
            control: External control, active on :math:`\lvert 1\rangle`.
            qreg: Register reflected about :math:`\lvert 0^n\rangle` when
                ``control`` is active.

        """
        cntrl_reflection_box(control, qreg, self.cnx_method)
