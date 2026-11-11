"""Basic fanout operation for quantum computing."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import cx, qubit

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
def _fanout_basic_cntrl_compose[n_data_qubits: nat, n_controls: nat](
    controls: array[qubit, n_controls], data_qs: array[qubit, n_data_qubits]
) -> None:
    """Compute the combined control predicate once, then apply the CX loop."""
    operation = MultiCntrlFunction[n_controls, array[qubit, n_data_qubits]](
        _fanout_basic_kernel[n_data_qubits], cnx[n_controls]
    )
    operation.compose(controls, data_qs)


@guppy.comptime
def _fanout_basic_cntrl[n_data_qubits: nat, k: nat](
    c: qubit, data_qs: array[qubit, n_data_qubits], controls: array[qubit, k]
) -> None:
    """Combine the k external controls with c, preserving the scalar API.

    Comptime packing retains references to the borrowed controls. The composer
    applies MultiCntrlFunction to these k + 1 controls and the original CX
    kernel, then returns every control and data qubit to its caller.
    """
    combined_controls = [*controls, c]
    _fanout_basic_cntrl_compose(combined_controls, data_qs)


@guppy.unitary
class fanout_basic[n_data_qubits: nat]:  # noqa: D101
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
        c: qubit,
        data_qs: array[qubit, n_data_qubits],
        controls: array[qubit, k],
    ) -> None:
        """Apply CX fanout under the AND of the external and intrinsic controls."""
        _fanout_basic_cntrl(c, data_qs, controls)

    @guppy
    @no_type_check
    def ctrl_daggered[k: nat](
        c: qubit,
        data_qs: array[qubit, n_data_qubits],
        controls: array[qubit, k],
    ) -> None:
        """Reuse controlled fanout because it is self-adjoint."""
        _fanout_basic_cntrl(c, data_qs, controls)
