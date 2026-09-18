"""The HHL algorithm for solving linear systems of equations [1].

References:
    [1] Harrow, A. W., Hassidim, A., & Lloyd, S. (2009). Quantum algorithm for linear
    systems of equations. Physical review letters, 103(15), 150502.

"""

from __future__ import annotations

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, nat, Function
from guppylang.std.quantum import h, qubit

from guppyalgos.primitives.subroutines.qft import iqft, qft
from guppyalgos.utils import transversal


@guppy
@no_type_check
def hhl[n_clock: nat, SystemReg](
    system_reg: SystemReg,
    clock_reg: array[qubit, n_clock],
    ancilla: qubit,
    power_oracle: Function[[qubit, SystemReg, int], None],
    eigenvalue_inversion: Function[[array[qubit, n_clock], qubit], None],
) -> None:
    """Circuit construction for the HHL algorithm.

    Args:
        system_reg: The quantum register which stores the state vector of the linear
            system. Should be initialized to the quantum state representing the input
            vector for the linear system.
        clock_reg: The clock register used for phase estimation.
        ancilla: The ancilla qubit used for eigenvalue inversion. Should be initialized
            to zero, and can be used as a flag qubit for amplitude amplification.
        power_oracle: The power oracle for QPE, performing controlled Hamiltonian
            simulation on the input matrix.
        eigenvalue_inversion: The conditional rotation implementing eigenvalue
            inversion.

    Returns:
        None

    """
    transversal(h, clock_reg)
    for n_index in range(n_clock):
        power_oracle(clock_reg[n_index], system_reg, 2**n_index)
    iqft(clock_reg)
    eigenvalue_inversion(clock_reg, ancilla)
    qft(clock_reg)
    for n_index in range(n_clock):
        power_oracle(clock_reg[n_index], system_reg, -(2**n_index))
    transversal(h, clock_reg)
