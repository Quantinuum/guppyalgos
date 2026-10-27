"""CCX V chains.

A CCX V chain is a CCX ladder followed by its own inverse without the last
Toffoli.

"""

import math
from itertools import pairwise
from typing import no_type_check

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.quantum import cx, qubit, toffoli, x

from guppyalgos.primitives.gate_decompositions.cnx.cnx_cca import cnx_cca_logdepth_dirty
from guppyalgos.primitives.subroutines.ladders.cnx_ladder import (
    cnx_ladder_logdepth,
    cnx_ladder_logdepth_num_ancilla,
)
from guppyalgos.primitives.subroutines.ladders.toffoli_ladder import (
    ToffoliLadderLog,
    log_toffoli_ladder_num_ancilla,
)


@guppy
@no_type_check
def ccx_v_chain[n: nat](
    controls_a: array[qubit, n], controls_b: array[qubit, n], target: qubit
) -> None:
    """Apply the CCX V chain in linear depth with 2n - 1 Toffoli gates.

    Args:
        controls_a (array[qubit, n]): Chain register, restored at the end.
        controls_b (array[qubit, n]): Second control of each Toffoli.
        target (qubit): Qubit flipped by the middle Toffoli.

    """
    if n == 0:
        return
    for i in range(n - 1):
        toffoli(controls_a[i], controls_b[i], controls_a[i + 1])
    toffoli(controls_a[n - 1], controls_b[n - 1], target)
    if n >= 2:
        for i in range(n - 2, -1, -1):
            toffoli(controls_a[i], controls_b[i], controls_a[i + 1])


def _blocks(n_pairs: int) -> tuple[int, int, list[int]]:
    """Block size, block count and block boundaries of the block split."""
    size = math.isqrt(n_pairs - 1) + 1 if n_pairs > 1 else 1
    n_blocks = -(-n_pairs // size)
    bounds = [0, n_pairs - size * (n_blocks - 1)]
    while bounds[-1] < n_pairs:
        bounds.append(bounds[-1] + size)
    return size, n_blocks, bounds


def _cnx_ladder_registers(blocks_b, cnx_targets) -> tuple[list, list]:
    """Build the two control registers of the cnx ladder."""
    n_blocks = len(blocks_b)
    controls_a = [blocks_b[-2][-1]] + [
        cnx_targets[j] for j in range(n_blocks - 2, 0, -1)
    ]
    controls_b = [list(blocks_b[-1])]
    for j in range(n_blocks - 3, -1, -1):
        controls_b.append([*blocks_b[j + 1][:-1], blocks_b[j][-1]])
    return controls_a, controls_b


def _apply_block_ladders(blocks_a, blocks_b, bottom_cca, inverse: bool = False) -> None:
    """Apply every block's CCX ladder."""
    n_blocks = len(blocks_a)
    for parity in (1, 0) if inverse else (0, 1):
        for j in range(n_blocks):
            n_gates = len(blocks_a[j]) - 1
            if (n_blocks - 1 - j) % 2 != parity or n_gates == 0:
                continue
            needed = log_toffoli_ladder_num_ancilla(2 * n_gates + 1)
            if needed == 0:
                gates = list(
                    zip(
                        blocks_a[j][:-1],
                        blocks_b[j][:-1],
                        blocks_a[j][1:],
                        strict=True,
                    )
                )
                for control_a, control_b, tgt in reversed(gates) if inverse else gates:
                    toffoli(control_a, control_b, tgt)
                continue
            borrowing = j + 1 < n_blocks
            cca = blocks_b[j + 1][:needed] if borrowing else bottom_cca[:needed]
            assert len(cca) == needed, "Not enough CCA qubits for the block ladder."
            if borrowing:
                for cca_qubit in cca:
                    x(cca_qubit)
            ladder_qs = [
                q
                for pair in zip(blocks_a[j][:-1], blocks_b[j][:-1], strict=True)
                for q in pair
            ]
            ladder_qs.append(blocks_a[j][-1])
            if inverse:
                ToffoliLadderLog().ascending_dagger_with_cca(ladder_qs, cca)  # ty: ignore[missing-argument]
            else:
                ToffoliLadderLog().ascending_with_cca(ladder_qs, cca)  # ty: ignore[missing-argument]
            if borrowing:
                for cca_qubit in cca:
                    x(cca_qubit)


def _apply_fan_in(
    control, blocks_a, blocks_b, cnx_targets, target, dirty, inverse: bool = False
) -> None:
    """Fan every block's middle Toffoli into the target."""
    n_blocks = len(blocks_a)

    def cx_fan_in_logdepth() -> None:
        """Apply a log-depth fan-in of the dirty qubits onto the target."""
        fold = []
        stride = 1
        while stride < len(dirty):
            for i in range(0, len(dirty) - stride, 2 * stride):
                fold.append((dirty[i + stride], dirty[i]))
            stride *= 2
        for src, dst in fold:
            cx(src, dst)
        toffoli(control, dirty[0], target)
        for src, dst in reversed(fold):
            cx(src, dst)

    def block_toffolis() -> None:
        """Apply one Toffoli per block, targeting the block's dirty qubit."""
        for j, dirty_qubit in enumerate(dirty):
            second = cnx_targets[j] if j < n_blocks - 1 else blocks_b[j][-1]
            toffoli(blocks_a[j][-1], second, dirty_qubit)

    steps = [cx_fan_in_logdepth, block_toffolis, cx_fan_in_logdepth, block_toffolis]
    for step in reversed(steps) if inverse else steps:
        step()


def _get_available_dirty_qubits(blocks_a, blocks_b) -> list:
    """List the qubits that can be borrowed as dirty workspace."""
    dirty_qubits = []
    for block_a, block_b in zip(blocks_a, blocks_b, strict=True):
        dirty_qubits += block_a[:-1] + block_b[:-1]
    return dirty_qubits


def _promise_register_size(n: int) -> int:
    """Size of the promise register for a chain over n pairs."""
    size, n_blocks, _ = _blocks(n)
    return (n_blocks - 1) + max(
        cnx_ladder_logdepth_num_ancilla(n_blocks - 1),
        log_toffoli_ladder_num_ancilla(2 * size - 1),
    )


def _apply_cntrl_ccx_v_chain(
    control, controls_a, controls_b, target, promise_register, inverse: bool = False
) -> None:
    """Apply the controlled CCX V chain on promise register qubits."""
    n_pairs = len(controls_a)
    size, n_blocks, bounds = _blocks(n_pairs)
    blocks_a = [controls_a[lo:hi] for lo, hi in pairwise(bounds)]
    blocks_b = [controls_b[lo:hi] for lo, hi in pairwise(bounds)]
    cnx_targets = promise_register[: n_blocks - 1]
    shared_cca = promise_register[n_blocks - 1 :]

    def apply_cnx_ladder(undo: bool = False) -> None:
        """Compute the AND of each block's controls_b onto cnx_targets."""
        if n_blocks < 2:
            return
        n_gates = n_blocks - 1
        controls_a, controls_b = _cnx_ladder_registers(blocks_b, cnx_targets)
        workspaces = [blocks_a[n_blocks - 1 - i] for i in range(n_gates)]
        dirty_a = [block[0] for block in workspaces]
        dirty_b = [block[1] for block in workspaces]
        if n_gates <= 3:
            targets = [*controls_a[1:], cnx_targets[0]]
            for i in reversed(range(n_gates)) if undo else range(n_gates):
                cnx_cca_logdepth_dirty(
                    [controls_a[i], *controls_b[i]], targets[i], dirty_a[i], dirty_b[i]
                )
        else:
            cnx_ladder_logdepth(n_gates, size + 1, undo)(
                controls_a,
                controls_b,
                cnx_targets[0],
                dirty_a,
                dirty_b,
                shared_cca[: cnx_ladder_logdepth_num_ancilla(n_gates)],
            )

    dirty = _get_available_dirty_qubits(blocks_a, blocks_b)[:n_blocks]
    assert len(dirty) == n_blocks, "Not enough dirty qubits for the fan-in."

    apply_cnx_ladder()
    _apply_block_ladders(blocks_a, blocks_b, shared_cca)
    _apply_fan_in(control, blocks_a, blocks_b, cnx_targets, target, dirty, inverse)
    _apply_block_ladders(blocks_a, blocks_b, shared_cca, inverse=True)
    apply_cnx_ladder(undo=True)


def _apply_ccx_v_chain(controls_a, controls_b, target) -> None:
    """Apply the CCX V chain, recursively."""
    n = len(controls_a)
    promise_register_size = next(
        (b for b in range(3, n - 2) if _promise_register_size(n - b) <= b), None
    )
    if promise_register_size is None:
        for i in range(n - 1):
            toffoli(controls_a[i], controls_b[i], controls_a[i + 1])
        toffoli(controls_a[n - 1], controls_b[n - 1], target)
        for i in range(n - 2, -1, -1):
            toffoli(controls_a[i], controls_b[i], controls_a[i + 1])
        return

    n_top = n - promise_register_size
    top_a, top_b = controls_a[:n_top], controls_b[:n_top]
    bottom_a, bottom_b = controls_a[n_top:], controls_b[n_top:]
    psi = bottom_a[0]

    def x_promise_register() -> None:
        """Apply X to every promise register qubit."""
        for promise_qubit in bottom_b:
            x(promise_qubit)

    cnx_cca_logdepth_dirty(bottom_b, psi, bottom_a[1], bottom_a[2])
    x_promise_register()
    _apply_cntrl_ccx_v_chain(psi, top_a, top_b, target, bottom_b)
    x_promise_register()
    cnx_cca_logdepth_dirty(bottom_b, psi, bottom_a[1], bottom_a[2])
    x_promise_register()
    _apply_cntrl_ccx_v_chain(psi, top_a, top_b, target, bottom_b, inverse=True)
    x_promise_register()
    _apply_ccx_v_chain(bottom_a, bottom_b, target)


def ccx_v_chain_logdepth(n: int) -> GuppyFunctionDefinition:
    """Build the CCX V chain in O(n) gates and O(log n) depth, ancilla free.

    Args:
        n: Number of (controls_a, controls_b) pairs.

    Returns:
        A Guppy function implementing the CCX V chain.

    Ref:
        Vivien Vandaele,
        "Asymptotically Optimal Quantum Circuits for Comparators and Incrementers",
        https://arxiv.org/abs/2603.12917

    """
    if n < 1:
        raise ValueError("The CCX V chain needs at least one pair")

    @guppy.comptime
    @no_type_check
    def ccx_v_chain_logdepth_impl(
        controls_a: array[qubit, comptime(n)],
        controls_b: array[qubit, comptime(n)],
        target: qubit,
    ) -> None:
        """Implement the CCX V chain in logarithmic depth.

        Args:
            controls_a (array[qubit, n]): First control of each Toffoli;
                also the target of the previous one.
            controls_b (array[qubit, n]): Second control of each Toffoli.
            target (qubit): target of the V chain.

        """
        _apply_ccx_v_chain(list(controls_a), list(controls_b), target)

    return ccx_v_chain_logdepth_impl
