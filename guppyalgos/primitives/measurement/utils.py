"""Utility functions for measurement and discarding of qubits."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, nat, nothing, owned, some
from guppylang.std.collections import Stack
from guppylang.std.option import Option
from guppylang.std.quantum import (
    Measurement,
    discard,
    discard_array,
    measure,
    qubit,
)


@guppy
@no_type_check
def measure_stack[n_work: nat](
    qs: Stack[qubit, n_work] @ owned,
) -> array[Option[Measurement], n_work]:  # ty: ignore[not-subscriptable]
    """Measure all qubits in a stack.

    Qubits are returned in the order they are popped off the stack.

    Args:
        qs (Stack[qubit, n_work] @ owned): The stack of qubits to measure.

    """
    meas_opts = array(nothing[Measurement]() for _ in range(n_work))
    for i in range(len(qs)):
        q = qs.pop()
        meas_opts[i].swap(some(measure(q))).unwrap_nothing()

    qs.discard_empty()
    return meas_opts


@guppy
@no_type_check
def discard_stack[n_work: nat](qs: Stack[qubit, n_work] @ owned) -> None:
    """Discard all qubits in a stack.

    Args:
        qs (Stack[qubit, n_work] @ owned): The stack of qubits to discard.

    """
    while len(qs) > 0:
        q = qs.pop()
        discard(q)

    qs.discard_empty()


@guppy
@no_type_check
def discard_nested_array[n_qubits_per_register: nat, n_registers: nat](
    nested_qregs: array[array[qubit, n_qubits_per_register], n_registers] @ owned,  # ty: ignore[not-subscriptable]
) -> None:
    """Discard all qubits in a nested array of registers.

    Args:
        nested_qregs: Nested array of qubit registers to discard.
        n_qubits_per_register (nat): Number of qubits in each register.
        n_registers (nat): Number of registers in the nested array.

    """
    for reg_idx in range(len(nested_qregs)):
        inner_qreg = nested_qregs.take(reg_idx)
        discard_array(inner_qreg)

    nested_qregs.discard_all_taken()
