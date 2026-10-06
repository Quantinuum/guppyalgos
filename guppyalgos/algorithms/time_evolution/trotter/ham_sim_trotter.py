"""Hamiltonian Simulation using Trotterization."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, nat, control, Unitary
from guppylang.std.quantum import qubit


@guppy.unitary
class ham_sim_trotter:  # noqa: D101
    @guppy
    @no_type_check
    def __call__[n_state_q: nat](
        state_qreg: array[qubit, n_state_q],
        trotter_step: Unitary[[array[qubit, n_state_q], float], None],
        n_steps: int,
        time_step: float,
    ) -> None:
        """Apply a full Hamiltonian simulation using Trotter steps."""
        for _ in range(n_steps):
            trotter_step(state_qreg, time_step)

    @guppy
    @no_type_check
    def controlled[n_state_q: nat, n_ctrl_q: nat](
        state_qreg: array[qubit, n_state_q],
        trotter_step: Unitary[[array[qubit, n_state_q], float], None],
        n_steps: int,
        time_step: float,
        controls: array[qubit, n_ctrl_q],
    ) -> None:
        """Apply a controlled Hamiltonian simulation using Trotter steps."""
        for _ in range(n_steps):
            with control(controls):
                trotter_step(state_qreg, time_step)
