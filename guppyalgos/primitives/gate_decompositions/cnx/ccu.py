"""CCU gates."""

from guppylang import guppy
from guppylang.std.builtins import array, control, nat
from guppylang.std.quantum import qubit, s, sdg, angle, toffoli, cx, ry
from .ccu_rel_phase import ccz_rel_phase, ccx_rel_phase, ccu_phase_correction
from .cnx import cnx
from .cnu import MultiCntrlFunctionParams


from typing import no_type_check


@guppy
@no_type_check
def ccx(control1: qubit, control2: qubit, target: qubit) -> None:
    """CCX gate implementation.

    .. code-block:: text

                                                                    ┌───┐
        control1: ───────────────────■─────────────────────■────■───┤ T ├───■──
                                     │                     │    │   └───┘   │
                                     │             ┌───┐   │  ┌─┴─┐┌─────┐┌─┴─┐
        control2: ───────■───────────┼─────────■───┤ T ├───┼──┤ X ├┤ Tdg ├┤ X ├─
                         │           │         │   └───┘   │  └───┘└─────┘└───┘
                  ┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌───┐
        target:   ┤ H ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ H ├────────
                  └───┘└───┘└─────┘└───┘└───┘└───┘└─────┘└───┘└───┘└───┘

    Args:
        target: The target qubit.
        control1: The first control qubit.
        control2: The second control qubit.

    """
    ccx_rel_phase(control1, control2, target)
    ccu_phase_correction(control1, control2)


@guppy
@no_type_check
def ccy(control1: qubit, control2: qubit, target: qubit) -> None:
    """CCY gate implementation.

    .. code-block:: text

                                                                         ┌───┐
        control1:  ——──────────────────────■─────—──────────────■────■───┤ T ├───■──
                                          │                     │    │   └───┘   │
                                          │             ┌───┐   │  ┌─┴─┐┌─────┐┌─┴─┐
        control2:  ───────────■───────────┼─────────■───┤ T ├───┼──┤ X ├┤ Tdg ├┤ X ├─
                              │           │         │   └───┘   │  └───┘└─────┘└───┘
                  ┌───┐┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌───┐┌─────┐
        target:   ┤ S ├┤ H ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ H ├┤ Sdg ├
                  └───┘└───┘└───┘└─────┘└───┘└───┘└───┘└─────┘└───┘└───┘└───┘└─────┘

    Args:
        target: The target qubit.
        control1: The first control qubit.
        control2: The second control qubit.

    """
    s(target)
    ccx(control1, control2, target)
    sdg(target)


@guppy
@no_type_check
def ccz(control1: qubit, control2: qubit, target: qubit) -> None:
    """CCZ gate implementation.

    .. code-block:: text

                                                               ┌───┐
        control1: ──────────────■─────────────────────■────■───┤ T ├───■──
                                │                     │    │   └───┘   │
                                │             ┌───┐   │  ┌─┴─┐┌─────┐┌─┴─┐
        control2: ──■───────────┼─────────■───┤ T ├───┼──┤ X ├┤ Tdg ├┤ X ├─
                    │           │         │   └───┘   │  └───┘└─────┘└───┘
                  ┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐
        target:   ┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├────────────
                  └───┘└─────┘└───┘└───┘└───┘└─────┘└───┘└───┘


    Args:
        target: The target qubit.
        control1: The first control qubit.
        control2: The second control qubit.

    """
    ccz_rel_phase(control1, control2, target)
    ccu_phase_correction(control1, control2)


@guppy
@no_type_check
def _ccry_toffoli_kernel(
    control1: qubit, control2: qubit, target: qubit, theta: angle
) -> None:
    """CCRy gate implementation using toffolis.

    Requires 2 toffolis and 2 ry. The intrinsic controls are preserved.

    .. code-block:: text


            control1: ─────────────■───────────────■───
                                   │               │
                                   │               │
            control2: ─────────────■───────────────■───
                                   │               │
                      ┌─────────┐┌─┴─┐┌─────────┐┌───┐
            target:   ┤ ry(t/2) ├┤ X ├┤ ry(-t/2)├┤ X ├─
                      └─────────┘└───┘└─────────┘└───┘

    Args:
        target: The target qubit.
        control1: The first control qubit.
        control2: The second control qubit.
        theta: The rotation angle.

    """
    ry(target, theta / 2)
    toffoli(control1, control2, target)
    ry(target, -theta / 2)
    toffoli(control1, control2, target)


@guppy
@no_type_check
def _ccry_cx_kernel(
    control1: qubit, control2: qubit, target: qubit, theta: angle
) -> None:
    """CCRy gate implementation using cx gates.

    Requires 4 cx and 4 ry. The intrinsic controls are preserved.

    .. code-block:: text

                       ┌────────────────────────────────┐x2
            control1: ─┼───────────────────────────■────┼───
                       │                           │    │
                       │                           │    │
            control2: ─┼────────────■──────────────┼────┼───
                       │            │              │    │
                       │ ┌───────┐┌─┴─┐┌────────┐┌─┴─┐  │
            target:   ─┼─┤ry(t/4)├┤ X ├┤ry(-t/4)├┤ X ├──┼───
                       │ └───────┘└───┘└────────┘└───┘  │
                       └────────────────────────────────┘

    Args:
        target: The target qubit.
        control1: The first control qubit.
        control2: The second control qubit.
        theta: The rotation angle.

    """
    # Preserve the decomposition's fixed two repetitions.
    ry(target, theta / 4)
    cx(control2, target)
    ry(target, -theta / 4)
    cx(control1, target)
    ry(target, theta / 4)
    cx(control2, target)
    ry(target, -theta / 4)
    cx(control1, target)


@guppy
@no_type_check
def _cry_kernel(ctrl: qubit, target: qubit, theta: angle) -> None:
    """Apply Ry under one control, preserving that control."""
    with control(ctrl):
        ry(target, theta)


@guppy
@no_type_check
def _ccry_cntrl_compose[n_controls: nat](
    controls: array[qubit, n_controls], target: qubit, theta: angle
) -> None:
    """Compute the joint predicate once and pass the angle to controlled Ry."""
    operation = MultiCntrlFunctionParams[n_controls, qubit, angle](
        _cry_kernel, cnx[n_controls]
    )
    operation.compose(controls, target, theta)


@guppy.comptime
def _ccry_cntrl[k: nat](
    control1: qubit,
    control2: qubit,
    target: qubit,
    theta: angle,
    controls: array[qubit, k],
) -> None:
    """Pack borrowed intrinsic controls with the external controls.

    Comptime packing preserves the scalar-qubit API without moving borrowed
    qubits. The ordinary Guppy composer applies the parameterized operation
    under the AND of all k + 2 controls and restores them automatically.
    """
    combined_controls = [*controls, control1, control2]
    _ccry_cntrl_compose(combined_controls, target, theta)


@guppy.unitary
class ccry_toffoli:  # noqa: D101
    @guppy
    @no_type_check
    def __call__(control1: qubit, control2: qubit, target: qubit, theta: angle) -> None:
        """Apply CCRy using the toffoli decomposition, preserving both controls."""
        _ccry_toffoli_kernel(control1, control2, target, theta)

    @guppy
    @no_type_check
    def daggered(control1: qubit, control2: qubit, target: qubit, theta: angle) -> None:
        """Apply the adjoint by reversing the rotation angle."""
        _ccry_toffoli_kernel(control1, control2, target, -theta)

    @guppy
    @no_type_check
    def controlled[k: nat](
        control1: qubit,
        control2: qubit,
        target: qubit,
        theta: angle,
        controls: array[qubit, k],
    ) -> None:
        """Apply controlled Ry using the AND of external and intrinsic controls."""
        _ccry_cntrl(control1, control2, target, theta, controls)

    @guppy
    @no_type_check
    def ctrl_daggered[k: nat](
        control1: qubit,
        control2: qubit,
        target: qubit,
        theta: angle,
        controls: array[qubit, k],
    ) -> None:
        """Reuse the joint-control operation with the inverse rotation angle."""
        _ccry_cntrl(control1, control2, target, -theta, controls)


@guppy.unitary
class ccry_cx:  # noqa: D101
    @guppy
    @no_type_check
    def __call__(control1: qubit, control2: qubit, target: qubit, theta: angle) -> None:
        """Apply CCRy using the cx decomposition, preserving both controls."""
        _ccry_cx_kernel(control1, control2, target, theta)

    @guppy
    @no_type_check
    def daggered(control1: qubit, control2: qubit, target: qubit, theta: angle) -> None:
        """Apply the adjoint by reversing the rotation angle."""
        _ccry_cx_kernel(control1, control2, target, -theta)

    @guppy
    @no_type_check
    def controlled[k: nat](
        control1: qubit,
        control2: qubit,
        target: qubit,
        theta: angle,
        controls: array[qubit, k],
    ) -> None:
        """Apply controlled Ry using the AND of external and intrinsic controls."""
        _ccry_cntrl(control1, control2, target, theta, controls)

    @guppy
    @no_type_check
    def ctrl_daggered[k: nat](
        control1: qubit,
        control2: qubit,
        target: qubit,
        theta: angle,
        controls: array[qubit, k],
    ) -> None:
        """Reuse the joint-control operation with the inverse rotation angle."""
        _ccry_cntrl(control1, control2, target, -theta, controls)
