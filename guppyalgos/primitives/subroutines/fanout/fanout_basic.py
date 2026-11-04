"""Basic fanout operation for quantum computing."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, nat, control
from guppylang.std.quantum import cx, qubit


@guppy(controllable=True)
@no_type_check
def _fanout_basic_kernel[n_data_qubits: nat](
    control: qubit,
    data_qs: array[qubit, n_data_qubits],
) -> None:
    """Apply basic fanout to one flat data register."""
    for i in range(n_data_qubits):
        cx(control, data_qs[i])


@guppy.unitary
class fanout_basic[n_data_qubits: nat]:  # noqa: D101 - Guppy unitary classes reject docstrings
    # Apply the circuit with dagger and external-control support.

    @guppy(controllable=True)
    @no_type_check
    def __call__(control: qubit, data_qs: array[qubit, n_data_qubits]) -> None:
        """Apply basic fanout to one flat data register."""
        _fanout_basic_kernel(control, data_qs)

    @guppy
    @no_type_check
    def daggered(control: qubit, data_qs: array[qubit, n_data_qubits]) -> None:
        """Apply the selected circuit mode."""
        _fanout_basic_kernel(control, data_qs)

    @guppy
    @no_type_check
    def ctrl_daggered[k: nat](
        c: qubit, data_qs: array[qubit, n_data_qubits], controls: array[qubit, k]
    ) -> None:
        """Apply the selected circuit mode."""
        with control(controls):
            _fanout_basic_kernel(c, data_qs)
