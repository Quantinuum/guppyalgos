"""CCU gates."""

from guppylang import guppy
from guppylang.std.quantum import qubit, s, sdg, angle, toffoli, cx, ry
from .ccu_rel_phase import ccz_rel_phase, ccx_rel_phase, ccu_phase_correction


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


@guppy(unitary=True)
@no_type_check
def ccry_toffoli(control1: qubit, control2: qubit, target: qubit, theta: angle) -> None:
    """CCRy gate implementation using toffolis.

    Requires 2 toffolis and 2 ry. Supports dagger and added external controls
    via Guppy modifiers; the two intrinsic controls remain separate.

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


@guppy(unitary=True)
@no_type_check
def ccry_cx(control1: qubit, control2: qubit, target: qubit, theta: angle) -> None:
    """CCRy gate implementation using cx gates.

    Requires 4 cx and 4 ry. Supports dagger and added external controls
    via Guppy modifiers; the two intrinsic controls remain separate.

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
    # Spell out the fixed two repetitions so Guppy can generate the adjoint.
    ry(target, theta / 4)
    cx(control2, target)
    ry(target, -theta / 4)
    cx(control1, target)
    ry(target, theta / 4)
    cx(control2, target)
    ry(target, -theta / 4)
    cx(control1, target)
