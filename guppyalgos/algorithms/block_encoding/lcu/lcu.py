"""LCU (Linear Combination of Unitaries) Guppy struct."""

from typing import no_type_check

from guppylang.decorator import guppy
from guppylang.std.builtins import Function


@guppy.struct
@no_type_check
class LCU[PrepRegs, TargetRegs]:
    r"""Guppy struct representing an LCU block encoding.

    Given PREPARE, SELECT, and UNPREPARE oracles, ``compose`` applies

    $$
    L = \mathrm{UNPREPARE}\,\mathrm{SELECT}\,\mathrm{PREPARE}.
    $$

    PREPARE initializes ``PrepRegs`` with the amplitudes used to select terms in
    the linear combination. SELECT applies the corresponding operation to
    ``TargetRegs``, controlled by the state of ``PrepRegs``. UNPREPARE then
    restores the initial ``PrepRegs`` basis state; it is typically the inverse
    of PREPARE, giving
    :math:`L = \mathrm{PREPARE}^{\dagger}\,\mathrm{SELECT}\,\mathrm{PREPARE}`.

    When these oracles encode an operator :math:`A` with normalization
    :math:`\lambda`, projecting ``PrepRegs`` onto its all-zero state extracts
    the block :math:`A / \lambda`.

    ``PrepRegs`` and ``TargetRegs`` are generic register types. They may be
    individual qubit arrays or Guppy structs containing multiple registers, as
    long as the three oracle signatures agree with the types used by ``LCU``.

    Type Parameters:
        PrepRegs: PREPARE register type shared by all three oracles.
        TargetRegs: Type of the target registers acted on by SELECT.

    Attributes:
        prepare: PREPARE oracle acting on ``PrepRegs``.
        select: SELECT oracle acting on ``PrepRegs`` and ``TargetRegs``.
        unprepare: UNPREPARE oracle restoring the initial ``PrepRegs`` state.

    """

    prepare: Function[[PrepRegs], None]
    select: Function[[PrepRegs, TargetRegs], None]
    unprepare: Function[[PrepRegs], None]

    @guppy
    @no_type_check
    def compose(
        self,
        prep_register: PrepRegs,
        state_register: TargetRegs,
    ) -> None:
        r"""Apply the LCU block encoding :math:`L`.

        Args:
            prep_register: The PREPARE register.
            state_register: The SELECT register.

        """
        self.prepare(prep_register)
        self.select(prep_register, state_register)
        self.unprepare(prep_register)


@guppy.struct
@no_type_check
class LCUCntrl[ControlRegs, PrepRegs, TargetRegs]:
    """LCU block encoding with an externally controlled SELECT oracle.

    ``ControlRegs`` is forwarded unchanged to ``cntrl_select`` and may be
    a qubit, qubit array, or structured register type. The supplied SELECT
    oracle defines the condition under which its target operation is applied.

    """

    prepare: Function[[PrepRegs], None]
    cntrl_select: Function[[ControlRegs, PrepRegs, TargetRegs], None]
    unprepare: Function[[PrepRegs], None]

    @guppy
    @no_type_check
    def compose(
        self,
        controls: ControlRegs,
        prep_register: PrepRegs,
        target_registers: TargetRegs,
    ) -> None:
        """Apply PREPARE, controlled SELECT, and UNPREPARE."""
        self.prepare(prep_register)
        self.cntrl_select(controls, prep_register, target_registers)
        self.unprepare(prep_register)
