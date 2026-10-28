"""Qubitization walk operator Guppy struct."""

from typing import no_type_check

from guppylang.decorator import guppy
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.quantum import qubit
from guppyalgos.algorithms.block_encoding.lcu.lcu import LCUCntrl, LCU
from guppyalgos.primitives.subroutines.reflection.reflection_box import (
    ReflectionCntrl,
    Reflection,
)


@guppy.comptime
def _assert_positive_power(power: nat @ comptime) -> None:
    if power <= 0:
        raise ValueError(f"Power must be a positive integer, got {power}.")


@guppy.struct
@no_type_check
class Qubitization[n_prep_q: nat, TargetRegs]:
    r"""Guppy struct representing the qubitization walk operator.

    The walk operator is

    $$
    W = R L,
    $$

    where :math:`L` is an LCU block encoding and :math:`R` is a reflection
    about the all-zero state of the PREPARE register. ``compose`` applies
    :math:`L` first and then :math:`R`, matching the right-to-left operator
    ordering above. The target registers are acted on by :math:`L`, while
    :math:`R` acts only on the PREPARE register.

    For a Hermitian operator :math:`H` block encoded as :math:`H / \lambda`,
    projecting the PREPARE register of :math:`W^k` onto the all-zero state gives
    the Chebyshev polynomial :math:`T_k(H / \lambda)`.

    Currently, ``block_encoding`` must be an :class:`LCU` block encoding.
    This constraint should be relaxed in the future to support arbitrary block
    encodings.

    Type Parameters:
        n_prep_q: Number of qubits in the PREPARE register.
        n_ctrl_q: Number of controls used by the reflection implementation.
        TargetRegs: Type of the target registers accepted by the block encoding.

    Attributes:
        block_encoding: LCU block encoding :math:`L`.
        reflection: Reflection :math:`R` acting on the PREPARE register.

    """

    block_encoding: LCU[
        array[qubit, n_prep_q],
        TargetRegs,
    ]  # TODO: generalize to arbitrary block encodings
    reflection: Reflection[n_prep_q]

    @guppy
    @no_type_check
    def compose(
        self,
        prep_register: array[qubit, n_prep_q],
        target_registers: TargetRegs,
    ) -> None:
        r"""Apply the qubitization walk operator :math:`W = R L`.

        Args:
            prep_register: The PREPARE register.
            target_registers: The target registers acted on by SELECT.

        """
        self.block_encoding.compose(prep_register, target_registers)
        self.reflection.compose(prep_register)

    @guppy
    @no_type_check
    def power(
        self,
        prep_register: array[qubit, n_prep_q],
        target_registers: TargetRegs,
        power: nat @ comptime,
    ) -> None:
        r"""Apply :math:`W^k`, where :math:`k` is the given ``power``.

        Args:
            prep_register: The PREPARE register.
            target_registers: The target registers acted on by SELECT.
            power: The power to which the walk operator is raised.

        """
        _assert_positive_power(power)

        for _ in range(power):
            self.compose(prep_register, target_registers)


@guppy.struct
@no_type_check
class QubitizationCntrl[n_prep_q: nat, TargetRegs]:
    r"""Externally controlled qubitization walk operator.

    PREPARE and UNPREPARE are applied unconditionally, while SELECT and the
    reflection are controlled. Thus the operation is the identity when the
    external control is zero and applies :math:`W = RL` when it is one.

    The reflection convention used here is
    :math:`R = I - 2\lvert 0\rangle\!\langle 0\rvert`. Consequently, projecting
    the PREPARE register of :math:`W^k` onto zero gives
    :math:`(-1)^k T_k(H / \lambda)`. Unlike for uncontrolled qubitization, this
    factor is a relative phase between the external-control branches and is
    therefore observable.

    """

    cntrl_block_encoding: LCUCntrl[
        qubit,
        array[qubit, n_prep_q],
        TargetRegs,
    ]  # TODO: generalize to arbitrary block encodings
    controlled_reflection: ReflectionCntrl[n_prep_q]

    @guppy
    @no_type_check
    def compose(
        self,
        control: qubit,
        prep_register: array[qubit, n_prep_q],
        target_registers: TargetRegs,
    ) -> None:
        """Apply one controlled qubitization walk step."""
        self.cntrl_block_encoding.compose(control, prep_register, target_registers)
        self.controlled_reflection.compose(control, prep_register)

    @guppy
    @no_type_check
    def power(
        self,
        control: qubit,
        prep_register: array[qubit, n_prep_q],
        target_registers: TargetRegs,
        power: nat @ comptime,
    ) -> None:
        """Apply a positive power of the controlled walk operator."""
        _assert_positive_power(power)

        for _ in range(power):
            self.compose(control, prep_register, target_registers)
