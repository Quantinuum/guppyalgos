"""Fanout based on log depth cx ladder."""

from typing import no_type_check

from guppylang import array, guppy, qubit
from guppylang.std.builtins import control, nat
from guppylang.std.quantum import cx

from guppyalgos.primitives.subroutines.ladders import CXLadderLog


@guppy
@no_type_check
def _fanout_log_controlled[n_ctrl: nat, n_data_qubits: nat](
    input: qubit,
    data_qs: array[qubit, n_data_qubits],
    extra_controls: array[qubit, n_ctrl],
) -> None:
    ladder = CXLadderLog()
    ladder.ascending_dagger(data_qs)
    with control(extra_controls):
        cx(input, data_qs[0])
    ladder.ascending(data_qs)


@guppy.unitary
class fanout_log[n_data_qubits: nat]:
    """Apply log depth fanout to one flat data register."""

    @guppy
    @no_type_check
    def __call__(  # noqa: D102
        control: qubit,
        data_qs: array[qubit, n_data_qubits],
    ) -> None:
        ladder = CXLadderLog()
        ladder.ascending_dagger(data_qs)
        cx(control, data_qs[0])
        ladder.ascending(data_qs)

    @guppy
    @no_type_check
    def daggered(  # noqa: D102
        control: qubit,
        data_qs: array[qubit, n_data_qubits],
    ) -> None:
        fanout_log(control, data_qs)

    @guppy
    @no_type_check
    def controlled[n_ctrl: nat](  # noqa: D102
        control: qubit,
        data_qs: array[qubit, n_data_qubits],
        extra_controls: array[qubit, n_ctrl],
    ) -> None:
        _fanout_log_controlled(control, data_qs, extra_controls)

    @guppy
    @no_type_check
    def ctrl_daggered[n_ctrl: nat](  # noqa: D102
        control: qubit,
        data_qs: array[qubit, n_data_qubits],
        extra_controls: array[qubit, n_ctrl],
    ) -> None:
        _fanout_log_controlled(control, data_qs, extra_controls)
