"""Measurement-based fanout implementations."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.quantum import (
    collect_measurements,
    cz,
    h,
    measure_array,
    project_z,
    qubit,
    reset,
    x,
    z,
)

from guppyalgos.primitives.subroutines.parity import parity_laqcc
from guppyalgos.utils import qarray, transversal


def fanout_measurement_compute_total_qubits(n_state_qubits: int) -> int:
    """Return the peak qubit count required by fanout compute.

    The count includes one control qubit, ``n_state_qubits`` supplied target
    qubits, and ``n_state_qubits - 2`` equal-check ancillas when at least three
    copies are requested.

    Args:
        n_state_qubits: Number of zero-initialized copy qubits to produce.

    Returns:
        Total number of target, copy, and equal-check qubits.

    """
    return 1 + n_state_qubits + max(n_state_qubits - 2, 0)


@guppy
@no_type_check
def fanout_measurement_compute[n_state_qubits: nat](
    q_target: qubit,
    qs_target_copies: array[qubit, n_state_qubits],
) -> None:
    """Fan out a target qubit into a supplied zero-state register.

    The circuit implements ``|y>|0>|0> -> |y>|y>|y>`` using measurement-assisted
    computation and feed-forward.

    For fewer than three copies, sequential controlled operations are used.
    Larger registers use measurement-assisted fanout with
    ``n_state_qubits - 2`` ancillas.

    It is up to the user to generate the copy register.

    Args:
        q_target: Qubit whose computational-basis value is copied.
        qs_target_copies: Caller-provided target register initialized to zero.

    """
    if comptime(n_state_qubits < 3):
        transversal(h, qs_target_copies)
        for i in range(n_state_qubits):
            cz(q_target, qs_target_copies[i])
        transversal(h, qs_target_copies)
        return

    n_ancilla_q: nat = comptime(max(n_state_qubits - 2, 0))
    qs_equal_checks = qarray(comptime(max(n_state_qubits - 2, 0)))

    transversal(h, qs_target_copies)
    transversal(h, qs_equal_checks)

    cz(q_target, qs_target_copies[0])
    for i in range(n_ancilla_q):
        cz(qs_equal_checks[i], qs_target_copies[i + 1])

    cz(q_target, qs_equal_checks[0])
    for i in range(1, n_ancilla_q):
        cz(qs_target_copies[i], qs_equal_checks[i])
    cz(
        qs_target_copies[n_state_qubits - 2],
        qs_target_copies[n_state_qubits - 1],
    )

    h(qs_target_copies[0])
    transversal(h, qs_equal_checks)

    bits_ab = measure_array(qs_equal_checks)

    bit_ab = False
    for i in range(n_ancilla_q):
        bit_ab ^= bits_ab[i].read()
        if bit_ab:
            x(qs_target_copies[i + 1])

    if bit_ab:
        z(qs_target_copies[n_state_qubits - 1])
    h(qs_target_copies[n_state_qubits - 1])


@guppy
@no_type_check
def fanout_measurement_uncompute[n_state_qubits: nat](
    q_target: qubit,
    qs_copies: array[qubit, n_state_qubits],
) -> None:
    """Uncompute a fanout copy register back to zero in place.

    The circuit implements ``|y>|y>|y> -> |y>|0>|0>`` using
    measurement-assisted computation
    and feed-forward.

    This is the inverse cleanup operation for
    :func:`fanout_measurement_compute`. It non-destructively measures the copy
    register, corrects the target using the measurement parity, and resets the
    supplied copies to zero.

    It is up to the user to discard the copy register after this operation.

    Args:
        q_target: Original target qubit used to create the copies.
        qs_copies: Caller-owned copy register to restore to zero.

    """
    transversal(h, qs_copies)

    measurements = array(project_z(qs_copies[i]) for i in range(n_state_qubits))
    bits = collect_measurements(measurements)

    bit_a = False
    for i in range(n_state_qubits):
        bit_a ^= bits[i]
    if bit_a:
        z(q_target)
    transversal(reset, qs_copies)


def fanout_measurement_parity_total_qubits(n_state_qubits: int) -> int:
    """Return the qubit count required by parity-based measurement fanout.

    The count includes one control qubit, ``n_state_qubits`` target qubits,
    and two LAQCC ancilla registers of size ``n_state_qubits - 3`` when the
    target register contains at least four qubits.

    Args:
        n_state_qubits: Number of fanout target qubits.

    Returns:
        Total number of control, target, and ancilla qubits.

    """
    return n_state_qubits + 1 + 2 * max(n_state_qubits - 3, 0)


@guppy
@no_type_check
def fanout_measurement_parity[n_state_qubits: nat](
    q_target: qubit, qs_inputs: array[qubit, n_state_qubits]
) -> None:
    """Apply parity-based measurement-assisted fanout to a target register.

    Hadamard conjugation reverses the direction of the CNOTs represented by
    the parity circuit. The operation therefore maps
    ``|c>|x_0, ..., x_n-1>`` to
    ``|c>|x_0 XOR c, ..., x_n-1 XOR c>``. This operation acts on an
    arbitrary input state. For four or more targets, the underlying LAQCC
    parity circuit allocates two ancilla registers, each containing
    ``n_state_qubits - 3`` qubits. Smaller registers use no ancillas.

    Args:
        q_target: Control qubit whose value is fanned out.
        qs_inputs: Target register. Its length determines the circuit and
            ancilla sizes at compile time.

    Examples:
        With control ``|1>`` and three targets in ``|0, 0, 0>``, the output
        targets are ``|1, 1, 1>`` and the control remains ``|1>``.

    """
    h(q_target)
    transversal(h, qs_inputs)
    parity_laqcc(q_target, qs_inputs)
    transversal(h, qs_inputs)
    h(q_target)
