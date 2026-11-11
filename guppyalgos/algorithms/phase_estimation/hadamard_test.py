"""Hadamard test primitive."""

from guppylang.std.builtins import Function

from typing import no_type_check

from guppylang import guppy
from guppylang.std.quantum import h, qubit


@guppy
@no_type_check
def hadamard_test[UnitaryRegs](
    ancilla: qubit,
    unitary_regs: UnitaryRegs,
    controlled_unitary: Function[[qubit, UnitaryRegs], None],
) -> None:
    """Apply the Hadamard test for a controlled unitary.

    The caller supplies the control qubit, the system register, and a
    controlled unitary acting on that register. The ancilla is prepared in
    ``|+>`` and returned through a final Hadamard so the usual Hadamard-test
    readout can be measured by the caller.

    Args:
        ancilla: Control qubit used for the Hadamard test.
        unitary_regs: The target unitary register, which may be an array or
            a structured register wrapper.
        controlled_unitary: Controlled oracle acting on ``(ancilla, unitary_regs)``.

    """
    h(ancilla)
    controlled_unitary(ancilla, unitary_regs)
    h(ancilla)
