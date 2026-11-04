"""Basic fanout operation for quantum computing."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.mem import mem_swap
from guppylang.std.quantum import cx, discard, qubit

from guppyalgos.primitives.gate_decompositions.cnx.cnu import MultiCntrlFunction
from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx
from guppyalgos.utils.guppy.array import join_arrays, split_array
from guppyalgos.utils.guppy.unsafe_borrow import (
    _unsafe_array_borrow,
    _unsafe_array_unborrow,
)


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
    """Compute one shared flag for the external and intrinsic controls.

    MultiCntrlFunction computes their AND, fans the flag out using CX gates,
    and uncomputes the flag. CNX work therefore scales with the control count
    once, rather than being repeated for every data qubit.
    """
    if comptime(k == 0 or n_data_qubits == 0):
        _fanout_basic_kernel(c, data_qs)
        return

    # Temporarily own the intrinsic control so it can join the control register.
    intrinsic = qubit()
    mem_swap(intrinsic, c)
    combined = join_arrays(
        _unsafe_array_borrow(controls), array(intrinsic), comptime(k + 1)
    )

    operation = MultiCntrlFunction[comptime(k + 1), array[qubit, n_data_qubits]](
        _fanout_basic_kernel, cnx
    )
    operation.compose(combined, data_qs)

    external, intrinsic_register = split_array(combined, k, 1)
    _unsafe_array_unborrow(controls, external)
    intrinsic = intrinsic_register.take(0)
    intrinsic_register.discard_all_taken()
    mem_swap(intrinsic, c)
    discard(intrinsic)


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
        """Fan out a shared AND flag for all controls."""
        _fanout_basic_controlled(c, data_qs, controls)

    @guppy
    @no_type_check
    def ctrl_daggered[k: nat](
        c: qubit, data_qs: array[qubit, n_data_qubits], controls: array[qubit, k]
    ) -> None:
        """Reuse the shared-flag circuit because fanout is self-adjoint."""
        _fanout_basic_controlled(c, data_qs, controls)
