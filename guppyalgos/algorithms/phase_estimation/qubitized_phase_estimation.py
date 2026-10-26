"""Canonical phase-estimation registers for qubitization."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import Function, array, nat
from guppylang.std.quantum import qubit


@guppy.struct
class QubitizationRegs[n_prep_q: nat, TargetRegs]:
    """Registers acted on by a qubitized walk operator during phase estimation."""

    prep: array[qubit, n_prep_q]
    target: TargetRegs


@guppy
@no_type_check
def qubitized_power_oracle[n_prep_q: nat, n_target_q: nat](
    control: qubit,
    regs: QubitizationRegs[
        n_prep_q,
        array[qubit, n_target_q],  # ty: ignore[not-subscriptable]
    ],
    power: int,
    cntrl_walk: Function[
        [qubit, array[qubit, n_prep_q], array[qubit, n_target_q]], None
    ],
) -> None:
    """Apply a controlled qubitization walk repeatedly as requested by QPE.

    Modifier migration is deferred: Guppy 1.1.1 cannot infer a ``Unitary``
    callback protocol depending on generic register sizes. Concrete walk and
    public struct integration remain deferred; no modifiers are advertised.
    """
    for _ in range(power):
        cntrl_walk(control, regs.prep, regs.target)
