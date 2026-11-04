"""Basic fanout operation for quantum computing."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.quantum import cx, discard, qubit, toffoli

from guppyalgos.primitives.gate_decompositions.cnx.cnu import MultiCntrlFunction
from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx


@guppy
@no_type_check
def _fanout_basic_kernel[n_data_qubits: nat](
    control: qubit,
    data_qs: array[qubit, n_data_qubits],
) -> None:
    """Apply basic fanout to one flat data register."""
    for i in range(n_data_qubits):
        cx(control, data_qs[i])


@guppy
@no_type_check
def _fanout_basic_controlled[n_data_qubits: nat, k: nat](
    c: qubit, data_qs: array[qubit, n_data_qubits], controls: array[qubit, k]
) -> None:
    """Compute an external-control flag, apply fanout, and uncompute the flag."""
    if comptime(k == 0 or n_data_qubits == 0):
        _fanout_basic_kernel(c, data_qs)
        return

    flag = qubit()
    compute_flag = MultiCntrlFunction[k, qubit](cx, cnx)
    compute_flag.compose(controls, flag)
    for i in range(n_data_qubits):
        toffoli(flag, c, data_qs[i])
    compute_flag.compose(controls, flag)
    discard(flag)


@guppy.unitary
class fanout_basic[n_data_qubits: nat]:  # noqa: D101 - Guppy unitary classes reject docstrings
    # Apply the circuit with dagger and external-control support.

    @guppy
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
    def controlled[k: nat](
        c: qubit, data_qs: array[qubit, n_data_qubits], controls: array[qubit, k]
    ) -> None:
        """Apply fanout under a shared external-control flag."""
        _fanout_basic_controlled(c, data_qs, controls)

    @guppy
    @no_type_check
    def ctrl_daggered[k: nat](
        c: qubit, data_qs: array[qubit, n_data_qubits], controls: array[qubit, k]
    ) -> None:
        """Reuse the shared-flag circuit because fanout is self-adjoint."""
        _fanout_basic_controlled(c, data_qs, controls)
