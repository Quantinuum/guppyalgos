"""Direct SwapUp operations."""

from math import ceil, log2
from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.quantum import qubit

from guppyalgos.utils.guppy.gates import _cswap


def _swapup_linear_schedule(
    n_index_qubits: int, n_state_qubits: int
) -> list[tuple[int, int, int]]:
    """Build the direct linear-scaling SwapUp controlled-swap schedule.

    Each tuple is ``(control_bit, lower, upper)``: if the little-endian index
    bit ``control_bit`` is set, the state items at ``lower`` and ``upper`` are
    swapped. The schedule moves the selected state item toward position 0 by
    applying larger strides before smaller strides.
    """
    if n_state_qubits < 2:
        raise ValueError("SwapUp requires at least two state qubits")

    expected_n_index_qubits = ceil(log2(n_state_qubits))
    if n_index_qubits != expected_n_index_qubits:
        raise ValueError(
            "SwapUp requires n_index_qubits == ceil(log2(n_state_qubits)); "
            f"got {n_index_qubits} index qubits for {n_state_qubits} state qubits"
        )

    schedule: list[tuple[int, int, int]] = []
    # Start with the most significant index bit so wide swaps happen first.
    for bit_pos in range(n_index_qubits - 1, -1, -1):
        stride = 2**bit_pos
        if bit_pos == n_index_qubits - 1:
            # Non-power-of-two registers have a shortened top swap layer.
            n_swaps = n_state_qubits - stride
        else:
            n_swaps = stride

        for lower in range(n_swaps):
            upper = lower + stride
            if upper >= n_state_qubits:
                raise ValueError(
                    "SwapUp schedule generated an out-of-range target: "
                    f"upper={upper}, n_state_qubits={n_state_qubits}"
                )
            schedule.append((bit_pos, lower, upper))

    return schedule


@guppy.comptime(daggerable=True)
@no_type_check
def _swapup_linear_qubit[n_index_qubits: nat, n_state_qubits: nat](
    index_qreg: array[qubit, n_index_qubits],
    state_qreg: array[qubit, n_state_qubits],
) -> None:
    for bit_pos, lower, upper in comptime(
        _swapup_linear_schedule(n_index_qubits, n_state_qubits)
    ):
        _cswap(index_qreg[bit_pos], state_qreg[lower], state_qreg[upper])


@guppy(daggerable=True)
@no_type_check
def swapup_linear_qubit[n_index_qubits: nat, n_state_qubits: nat](
    index_qreg: array[qubit, n_index_qubits],
    state_qreg: array[qubit, n_state_qubits],
) -> None:
    """Move ``state_qreg[x]`` to ``state_qreg[0]`` with the linear SwapUp.

    ``n_state_qubits`` must be at least two. The index register must contain
    ``ceil(log2(n_state_qubits))`` qubits, with ``index_qreg[0]`` holding the least
    significant bit. Values initially at other state-register positions may move; they
    are not left unchanged. For non-power-of-two registers, users must ensure no
    amplitude is present on invalid index values.
    """
    _swapup_linear_qubit(index_qreg, state_qreg)


@guppy.comptime(daggerable=True)
@no_type_check
def _swapup_linear_register[n_index_qubits: nat, width: nat, n_state_registers: nat](
    index_qreg: array[qubit, n_index_qubits],
    state_qregs: array[
        array[qubit, width],  # ty: ignore[not-subscriptable]
        n_state_registers,
    ],
) -> None:
    for bit_pos, lower, upper in comptime(
        _swapup_linear_schedule(n_index_qubits, n_state_registers)
    ):
        _cswap(index_qreg[bit_pos], state_qregs[lower], state_qregs[upper])


@guppy(daggerable=True)
@no_type_check
def swapup_linear_register[n_index_qubits: nat, width: nat, n_state_registers: nat](
    index_qreg: array[qubit, n_index_qubits],
    state_qregs: array[
        array[qubit, width],  # ty: ignore[not-subscriptable]
        n_state_registers,
    ],
) -> None:
    """Move ``state_qreg[x]`` to ``state_qreg[0]`` for little-endian ``x``.

    ``n_state_registers`` must be at least two. The index register must contain
    ``ceil(log2(n_state_registers))`` qubits, with ``index_qreg[0]`` holding
    the least significant bit. ``state_qregs`` contains ``n_state_registers``
    registers, each containing ``width`` qubits. At each scheduled step, the
    control qubit swaps ``state_qregs[lower][j]`` with ``state_qregs[upper][j]``
    for every qubit position ``j`` from ``0`` to ``width - 1``. For a
    non-power-of-two number of state registers, users must ensure no amplitude
    is present on invalid index values.
    """
    _swapup_linear_register(index_qreg, state_qregs)


@guppy.overload(swapup_linear_qubit, swapup_linear_register)
@no_type_check
def swapup_linear() -> None:
    """Move the index-selected qubit or state register to position zero.

    Either:
        swapup_linear(index_qreg, state_qreg)
    Or:
        swapup_linear(index_qreg, state_qregs)
    """
    ...
