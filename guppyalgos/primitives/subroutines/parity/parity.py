"""Sequential and measurement-assisted quantum parity circuits."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, comptime, nat, control
from guppylang.std.quantum import cz, h, measure_array, qubit, z

from guppyalgos.utils import qarray, transversal


def parity_laqcc_total_qubits(n_input_qubits: int) -> int:
    """Return the total qubit count required by LAQCC parity.

    The count includes ``n_input_qubits`` input qubits, one target qubit, and
    two ancilla registers of size ``n_input_qubits - 3`` when there are at
    least four inputs. The sequential fallback uses no ancilla qubits.

    Args:
        n_input_qubits: Number of input qubits whose parity is computed.

    Returns:
        Total number of input, target, and ancilla qubits.

    """
    return n_input_qubits + 1 + 2 * max(n_input_qubits - 3, 0)


@guppy(controllable=True)
@no_type_check
def _parity_sequential_kernel[n_qs_qubits: nat](
    q_target: qubit, qs_inputs: array[qubit, n_qs_qubits]
) -> None:
    """XOR the parity of the input register into a target qubit.

    This operation implements
    ``|x_0, ..., x_n-1>|t> -> |x_0, ..., x_n-1>|t XOR x_0 XOR ... XOR x_n-1>``
    using a sequence of controlled operations and no ancilla qubits.

    Args:
        q_target: Qubit into which the input parity is XORed.
        qs_inputs: Input register. Its length determines the circuit size at
            compile time.

    Examples:
        For inputs ``|1, 0, 1, 1>`` and target ``|0>``, the target becomes
        ``|1>`` because ``0 XOR 1 XOR 0 XOR 1 XOR 1 = 1``. The four input
        qubits are unchanged.

    """
    h(q_target)
    for i in range(n_qs_qubits):
        cz(q_target, qs_inputs[i])
    h(q_target)


@guppy.unitary
class parity_sequential[n_qs_qubits: nat]:  # noqa: D101 - Guppy unitary classes reject docstrings
    # Apply the circuit with dagger and external-control support.

    @guppy(controllable=True)
    @no_type_check
    def __call__(q_target: qubit, qs_inputs: array[qubit, n_qs_qubits]) -> None:
        """XOR the parity of the input register into a target qubit.

        This operation implements
        ``|x_0, ..., x_n-1>|t> -> |x_0, ..., x_n-1>|t XOR x_0 XOR ... XOR x_n-1>``
        using a sequence of controlled operations and no ancilla qubits.

        Args:
            q_target: Qubit into which the input parity is XORed.
            qs_inputs: Input register. Its length determines the circuit size at
                compile time.

        Examples:
            For inputs ``|1, 0, 1, 1>`` and target ``|0>``, the target becomes
            ``|1>`` because ``0 XOR 1 XOR 0 XOR 1 XOR 1 = 1``. The four input
            qubits are unchanged.

        """
        _parity_sequential_kernel(q_target, qs_inputs)

    @guppy
    @no_type_check
    def daggered(q_target: qubit, qs_inputs: array[qubit, n_qs_qubits]) -> None:
        """Apply the selected circuit mode."""
        _parity_sequential_kernel(q_target, qs_inputs)

    @guppy
    @no_type_check
    def ctrl_daggered[k: nat](
        q_target: qubit, qs_inputs: array[qubit, n_qs_qubits], controls: array[qubit, k]
    ) -> None:
        """Apply the selected circuit mode."""
        with control(controls):
            _parity_sequential_kernel(q_target, qs_inputs)


@guppy
@no_type_check
def parity_laqcc[n_qs_qubits: nat](
    q_target: qubit, qs_inputs: array[qubit, n_qs_qubits]
) -> None:
    """XOR input parity into a target using measurement-assisted computation.

    This operation implements
    ``|x_0, ..., x_n-1>|t> -> |x_0, ..., x_n-1>|t XOR x_0 XOR ... XOR x_n-1>``
    using a sequence of controlled operations and with ancilla qubits.

    This is a constant-depth implementation of the parity circuit,
    For fewer than four inputs, uses
    :func:`parity_sequential`. Otherwise, the operation uses two ancilla
    registers of size ``n_qs_qubits - 3`` and measurement feed-forward. The
    input qubits are unchanged in both cases.

    Args:
        q_target: Qubit into which the input parity is XORed.
        qs_inputs: Input register. Its length determines the circuit and
            ancilla sizes at compile time.

    Examples:
        Applied to inputs ``|1, 0, 1, 1>`` and target ``|0>``, this operation
        leaves the inputs unchanged and changes the target to ``|1>``.

    """
    if comptime(n_qs_qubits < 4):
        parity_sequential(q_target, qs_inputs)
        return

    n_ancilla_q = comptime(max(n_qs_qubits - 3, 0))
    qs_equal_checks = qarray(comptime(max(n_qs_qubits - 3, 0)))
    qs_target_copies = qarray(comptime(max(n_qs_qubits - 3, 0)))

    h(q_target)

    transversal(h, qs_equal_checks)
    transversal(h, qs_target_copies)

    cz(q_target, qs_inputs[0])
    for i in range(n_ancilla_q):
        cz(qs_equal_checks[i], qs_target_copies[i])

    cz(q_target, qs_equal_checks[0])
    for i in range(n_ancilla_q - 1):
        cz(qs_target_copies[i], qs_equal_checks[i + 1])
    cz(qs_target_copies[n_ancilla_q - 1], qs_inputs[n_qs_qubits - 1])

    cz(q_target, qs_inputs[1])
    for i in range(n_ancilla_q):
        cz(qs_target_copies[i], qs_inputs[i + 2])

    transversal(h, qs_equal_checks)
    transversal(h, qs_target_copies)

    bits_ab = measure_array(qs_equal_checks)
    bits_c = measure_array(qs_target_copies)

    # Classical feed-forward to apply Z gates based on measurement results

    bit_c = False
    for i in range(n_ancilla_q):
        bit_c ^= bits_c[i].read()

    if bit_c:
        z(q_target)

    h(q_target)

    bit_ab = False
    for i in range(n_ancilla_q):
        bit_ab ^= bits_ab[i].read()
        if bit_ab:
            z(qs_inputs[i + 2])
    if bit_ab:
        z(qs_inputs[n_qs_qubits - 1])
