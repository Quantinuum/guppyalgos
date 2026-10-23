"""Phase estimation."""

from typing import no_type_check
from guppylang.decorator import guppy
from guppylang.std.builtins import Unitary, array, control, nat, Function
from guppylang.std.quantum import qubit

from guppyalgos.primitives.subroutines.qft import iqft, qft


@guppy
@no_type_check
def qpe[
    n_q_a: nat,
    UnitaryRegs,
](
    phase_reg: array[qubit, n_q_a],
    unitary_regs: UnitaryRegs,
    power_oracle: Function[[qubit, UnitaryRegs, int], None],
    # TODO update to Callable when type bounds can be generic
) -> None:
    """Algorithmic primitive for canonical quantum phase estimation.

    Args:
        phase_reg: The phase register on which the phase is written.
        unitary_regs: Registers acted on by the powered controlled unitary.
            It is the responsibility of the powered unitary to interpret this
            set of registers correctly.
        power_oracle: Controlled oracle that applies the requested unitary power.

    Returns:
        The binary power phase kickback encoded ancilla register
        in superposition with the unitary registers.

    """
    for n_index in range(n_q_a):
        power_oracle(phase_reg[n_index], unitary_regs, 2**n_index)
    iqft(phase_reg)


@guppy
@no_type_check
def qpe_unitary[n_q_a: nat, UnitaryRegs, UnitaryArg](
    phase_reg: array[qubit, n_q_a],
    unitary_regs: UnitaryRegs,
    unitary: Unitary[[UnitaryRegs, UnitaryArg], None],
    unitary_arg: UnitaryArg,
) -> None:
    """Estimate a unitary's phase using its custom controlled modifier."""
    for n_index in range(n_q_a):
        for _ in range(2**n_index):
            with control(phase_reg[n_index]):
                unitary(unitary_regs, unitary_arg)
    iqft(phase_reg)


@guppy
@no_type_check
def iqpe[n_q_a: nat, UnitaryRegs](
    phase_reg: array[qubit, n_q_a],
    unitary_regs: UnitaryRegs,
    inverse_power_oracle: Function[[qubit, UnitaryRegs, int], None],
    # TODO update to Callable when type bounds can be generic
) -> None:
    """Algorithmic primitive for canonical quantum phase estimation.

    Args:
        phase_reg: The phase register on which the phase is written.
        unitary_regs: Registers acted on by the powered controlled unitary.
            It is the responsibility of the powered unitary to interpret this
            set of registers correctly.
        inverse_power_oracle: Controlled oracle for the inverse unitary powers.

    Returns:
        The binary power phase kickback encoded ancilla register
        in superposition with the unitary registers.

    """
    qft(phase_reg)
    for n_index in range(n_q_a):
        # note the inverse is contained in the function definition
        inverse_power_oracle(phase_reg[n_index], unitary_regs, 2**n_index)
