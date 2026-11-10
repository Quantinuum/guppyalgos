"""SelectSWAP data lookup implementation."""

from collections.abc import Callable
from math import ceil, log2
from typing import no_type_check

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, dagger, nat
from guppylang.std.quantum import qubit

from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and,
    uncompute_and,
)
from guppyalgos.primitives.subroutines.fanout import fanout_basic, fanout_from_data
from guppyalgos.algorithms.select.qrom import qrom_unary_iteration
from guppyalgos.primitives.subroutines.swap import swapup_linear


def selectswap[TargetRegs, n_i_q: nat, n_j_q: nat](
    data_inputs: list[list[bool]],
    n_target_registers: int,
    fanout_op: GuppyFunctionDefinition = fanout_basic,
    comp_and_op: GuppyFunctionDefinition[[qubit, qubit, qubit], None] = temp_and,
    uncomp_and_op: GuppyFunctionDefinition[[qubit, qubit, qubit], None] = uncompute_and,
    fanout_from_data_fn: Callable[
        [list[list[bool]], GuppyFunctionDefinition],
        GuppyFunctionDefinition[[qubit, TargetRegs], None],
    ] = fanout_from_data,
    uncompute: bool = False,
) -> GuppyFunctionDefinition:
    """Construct a forward or inverse SelectSWAP operation.

    Each data entry defines the Boolean pattern for one data operation ``D_l``. The
    entries are grouped into rows of ``n_target_registers`` entries. The returned
    operation uses a little-endian ``index_i_qreg`` to load one row into equal-width
    target registers and a little-endian ``index_j_qreg`` to move the selected target
    register to position zero. If the number of data entries is not divisible by
    ``n_target_registers``, the final row is padded with all-False data entries.

    ``index_i_qreg`` values that do not correspond to a QROM row load no data. SwapUp
    is still applied according to ``index_j_qreg``. For non-power-of-two
    ``n_target_registers``, callers must ensure no amplitude is present on
    ``index_j_qreg`` values greater than or equal to the number of target registers.

    Args:
        data_inputs: Flat non-empty list of equal-width Boolean data entries.
        n_target_registers: Number of physical target registers; must be at least two
            and produce at least three QROM rows.
        uncompute: Whether to construct the inverse operation.
        fanout_op: Flat fanout operation forwarded to unary-iteration QROM.
        comp_and_op: Temporary-AND compute operation forwarded to QROM.
        uncomp_and_op: Temporary-AND uncompute operation forwarded to QROM.
        fanout_from_data_fn: Python metafunction adapting a grouped data row and
            ``fanout_op`` to the nested target-register type.

    Returns:
        A Guppy operation accepting ``index_i_qreg``, ``index_j_qreg``, and the nested
        target-register array.

    """
    if not data_inputs:
        raise ValueError("SelectSWAP requires non-empty data_inputs")
    if not isinstance(n_target_registers, int) or isinstance(n_target_registers, bool):
        raise ValueError("SelectSWAP n_target_registers must be an integer")
    if n_target_registers < 2:
        raise ValueError("SelectSWAP requires at least two target registers")
    if any(
        not isinstance(data_entry, list)
        or not data_entry
        or any(type(bit) is not bool for bit in data_entry)
        for data_entry in data_inputs
    ):
        raise ValueError(
            "SelectSWAP requires each data entry to be a non-empty list of bool values"
        )

    width = len(data_inputs[0])
    if any(len(data_entry) != width for data_entry in data_inputs):
        raise ValueError("SelectSWAP requires all data entries to have the same width")

    n_data_entries = len(data_inputs)
    n_rows = ceil(n_data_entries / n_target_registers)
    if n_rows < 3:
        raise ValueError(
            "SelectSWAP requires at least three QROM rows; "
            f"got {n_rows} rows from {n_data_entries} data entries and "
            f"{n_target_registers} target registers"
        )

    n_index_i_qubits = ceil(log2(n_rows))
    n_index_j_qubits = ceil(log2(n_target_registers))
    grouped_data = [
        [
            data_inputs[index]
            if index < n_data_entries
            else [False for _ in range(width)]
            for index in range(
                row * n_target_registers,
                (row + 1) * n_target_registers,
            )
        ]
        for row in range(n_rows)
    ]
    qrom = qrom_unary_iteration(
        grouped_data,
        fanout_op=fanout_op,
        comp_and_op=comp_and_op,
        uncomp_and_op=uncomp_and_op,
        fanout_from_data_fn=fanout_from_data_fn,
    )

    if uncompute:

        @guppy
        @no_type_check
        def select_swap_uncompute_operation(
            index_i_qreg: array[qubit, n_index_i_qubits],
            index_j_qreg: array[qubit, n_index_j_qubits],
            target_qregs: array[
                array[qubit, width],
                n_target_registers,
            ],
        ) -> None:
            with dagger:
                swapup_linear(index_j_qreg, target_qregs)
            qrom(index_i_qreg, target_qregs)

        return select_swap_uncompute_operation

    @guppy
    @no_type_check
    def selectswap_operation(
        index_i_qreg: array[qubit, n_index_i_qubits],
        index_j_qreg: array[qubit, n_index_j_qubits],
        target_qregs: array[
            array[qubit, width],
            n_target_registers,
        ],
    ) -> None:
        qrom(index_i_qreg, target_qregs)
        swapup_linear(index_j_qreg, target_qregs)

    return selectswap_operation
