"""Compose a multi-controlled operation with a selectable CNX kernel."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import Function, array, nat
from guppylang.std.quantum import discard, qubit


@guppy.struct
@no_type_check
class MultiCntrlFunction[m: nat, TargetRegs]:
    """Apply a singly controlled operation using an AND of the controls.

    ``operation`` must preserve its control qubit. ``cnx_kernel`` must preserve
    the controls and be self-inverse on its input registers, leaving its internal
    ancillas unentangled with them. A randomized approximate kernel must replay
    the same randomness on
    both calls to uncompute the work qubit. Use at least one control.

    ``TargetRegs`` can be a qubit, an array, or a Guppy struct of registers.
    One control is passed directly to the operation without allocating work.
    Multiple controls use one clean work qubit in addition to kernel ancillas.
    """

    operation: Function[[qubit, TargetRegs], None]
    cnx_kernel: Function[[array[qubit, m], qubit], None]

    @guppy
    @no_type_check
    def compose(self, controls: array[qubit, m], targets: TargetRegs) -> None:
        """Compute the control predicate, apply the operation, and uncompute."""
        if m == 1:
            self.operation(controls[0], targets)
        else:
            work = qubit()
            self.cnx_kernel(controls, work)
            self.operation(work, targets)
            self.cnx_kernel(controls, work)
            discard(work)
