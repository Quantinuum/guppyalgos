"""CCU gates up to a relative phase."""

from guppylang import guppy
from guppylang.std.builtins import dagger
from guppylang.std.quantum import qubit, cx, t, tdg, h


from typing import no_type_check


@guppy(unitary=True)
@no_type_check
def ccz_rel_phase(
    control1: qubit,
    control2: qubit,
    target: qubit,
) -> None:
    """CCZ gate implementation up to a relative phase.

    Supports dagger and added external controls via Guppy modifiers.

    .. code-block:: text

        control1: ───────────────────■─────────────────────■────────────
                                     │                     │
        control2: ───────■───────────┼─────────■───────────┼────────────
                       ┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐
        target:   ─────┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├─────
                       └───┘└─────┘└───┘└───┘└───┘└─────┘└───┘└───┘

    Args:
        target: The target qubit.
        control1: The first control qubit.
        control2: The second control qubit.

    """
    cx(control2, target)
    tdg(target)
    cx(control1, target)
    t(target)
    cx(control2, target)
    tdg(target)
    cx(control1, target)
    t(target)


@guppy(unitary=True)
@no_type_check
def ccu_phase_correction(control1: qubit, control2: qubit) -> None:
    """Phase correction circuit for relative phase implementations of CCU gates.

    Supports dagger and added external controls via Guppy modifiers.

    .. code-block:: text

                             ┌───┐
        control1: ───────■───┤ T ├───■──
                         │   └───┘   │
                  ┌───┐┌─┴─┐┌─────┐┌─┴─┐
        control2: ┤ T ├┤ X ├┤ Tdg ├┤ X ├
                  └───┘└───┘└─────┘└───┘

    Args:
        control1: The first control qubit.
        control2: The second control qubit.

    """
    t(control2)
    cx(control1, control2)
    t(control1)
    tdg(control2)
    cx(control1, control2)


@guppy
@no_type_check
def ccx_rel_phase(control1: qubit, control2: qubit, target: qubit) -> None:
    """CCX gate implementation up to a relative phase.

    .. code-block:: text

        control1: ───────────────────■─────────────────────■────────────
                                     │                     │
        control2: ───────■───────────┼─────────■───────────┼────────────
                  ┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌───┐
        target:   ┤ H ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ H ├
                  └───┘└───┘└─────┘└───┘└───┘└───┘└─────┘└───┘└───┘└───┘

    Args:
        target: The target qubit.
        control1: The first control qubit.
        control2: The second control qubit.

    """
    h(target)
    ccz_rel_phase(control1, control2, target)
    h(target)


@guppy
@no_type_check
def ccx_rel_phase_dg(control1: qubit, control2: qubit, target: qubit) -> None:
    """CCX gate implementation up to a relative phase using the adjoint circuit.

    Args:
        target: The target qubit.
        control1: The first control qubit.
        control2: The second control qubit.

    """
    h(target)
    with dagger:
        ccz_rel_phase(control1, control2, target)
    h(target)
