"""Calculate the hamming weight of a qubit register."""

from math import ceil, log2
from typing import no_type_check

import numpy as np
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, comptime, frozenarray, owned, nat
from guppylang.std.mem import mem_swap
from guppylang.std.option import nothing, some
from guppylang.std.quantum import qubit

from guppyalgos.primitives.arithmetic.adder.bit_adders import (
    full_adder,
    full_adder_inverse,
    half_adder,
    half_adder_inverse,
)
from guppyalgos.primitives.measurement import discard_array
from guppyalgos.utils import qarray


def _adder_indices(n: int) -> tuple[list[list[int]], list[int]]:
    """Calculate the sequence of bit adders in hamming weight circuit.

    Produces a log-depth sequence of half and full adders, which add the bits
    of the input register in a tree until the total number of bits (the hamming
    weight) is stored on the bits corresponding to `indices_by_level`.

    Returns:
        ops: list of ops of the form [i,j,k] where these indicate the qubit indices into
        a full adder. When k=-1 this indicates a half-adder on i,j.
        indices_by_level: the final positions of the level-l indices which store the
        hamming weight after the sequence of adders have been performed.

    """
    bitlevels = np.zeros(n)
    ops: list[list[int]] = []
    limit = int(np.ceil(np.log2(n)))
    for level in range(limit):
        while True:
            indices = np.nonzero(bitlevels == level)[0].tolist()
            if len(indices) == 0:
                raise RuntimeError("bug in hamming weight index calculation")
            if len(indices) == 1:
                break
            if len(indices) == 2:
                i = indices[0]
                j = indices[1]
                ops.append([i, j, -1])
                bitlevels[j] += 1
                break
            # use as many full adders as possible
            num_threes = len(indices) // 3
            for t in range(num_threes):
                i = indices[3 * t]
                j = indices[3 * t + 1]
                k = indices[3 * t + 2]
                ops.append([i, j, k])
                bitlevels[i] = -1
                bitlevels[k] += 1

    max_lvl = int(np.max(bitlevels))
    indices_by_level: list[int] = []

    for lvl in range(max_lvl + 1):
        indices_by_level.append(int(np.nonzero(bitlevels == lvl)[0][0]))

    return ops, indices_by_level


@guppy
@no_type_check
def _full_adder_with_swaps[n_q: nat](
    q: array[qubit, n_q],
    ancilla: array[qubit, n_q],
    inds: frozenarray[int, 3],
    ancilla_index: int,
) -> None:
    """Bring in an ancilla, full-add and then swap out a junk qubit."""
    full_adder(q[inds[0]], q[inds[1]], q[inds[2]], ancilla[ancilla_index])
    mem_swap(q[inds[2]], q[inds[1]])
    mem_swap(q[inds[2]], ancilla[ancilla_index])


@guppy
@no_type_check
def _half_adder_with_swaps[n_q: nat](
    q: array[qubit, n_q],
    ancilla: array[qubit, n_q],
    inds: frozenarray[int, 3],
    ancilla_index: int,
) -> None:
    """Bring in an ancilla, half-add and then swap out a junk qubit."""
    half_adder(q[inds[0]], q[inds[1]], ancilla[ancilla_index])
    mem_swap(q[inds[1]], q[inds[0]])
    mem_swap(q[inds[1]], ancilla[ancilla_index])


@guppy
@no_type_check
def _full_adder_with_swaps_inv[n_q: nat](
    q: array[qubit, n_q],
    ancilla: array[qubit, n_q],
    inds: frozenarray[int, 3],
    ancilla_index: int,
) -> None:
    """Inverse of full adder with swaps."""
    mem_swap(q[inds[2]], ancilla[ancilla_index])
    mem_swap(q[inds[2]], q[inds[1]])
    full_adder_inverse(q[inds[0]], q[inds[1]], q[inds[2]], ancilla[ancilla_index])


@guppy
@no_type_check
def _half_adder_with_swaps_inv[n_q: nat](
    q: array[qubit, n_q],
    ancilla: array[qubit, n_q],
    inds: frozenarray[int, 3],
    ancilla_index: int,
) -> None:
    """Inverse of half adder with swaps."""
    mem_swap(q[inds[1]], ancilla[ancilla_index])
    mem_swap(q[inds[1]], q[inds[0]])
    half_adder_inverse(q[inds[0]], q[inds[1]], ancilla[ancilla_index])


def num_hamming_weight_bits(n: int) -> int:
    """Get number of bits required to represent the hamming weight of n bits."""
    nhambits = ceil(log2(n))
    if (n & (n - 1)) == 0:  # if n a power of 2 we need an extra bit for the all 1s case
        nhambits += 1
    return nhambits


def hamming_weight_func[n_q: nat, logn_q: nat, nmlogn_q: nat](
    n: int,
) -> GuppyFunctionDefinition[
    [array[qubit, n_q]],
    tuple[array[qubit, logn_q], array[qubit, nmlogn_q]],
]:
    """Produce a guppy function for computing the hamming weight of a register.

    The circuit takes in n qubits in `main_reg` and computes the hamming weight of
    `main_reg` onto roughly log2(n) qubits (see num_hamming_weight_bits) using n ancilla
    while the rest of the qubits from `main_reg` and `ancilla` become junk that must be
    uncomputed.

    The guppy function returns the hamming weight register and the junk register.

    UNSAFE:
        Leave main_reg argument in an invalid state, must be uncomputed using
        hamming_weight_func_inv.
    """
    nhambits = num_hamming_weight_bits(n)
    ops, idxs = _adder_indices(n)

    @guppy
    @no_type_check
    def hamm(
        main_reg: array[qubit, n],
    ) -> tuple[array[qubit, nhambits], array[qubit, comptime(2 * n - nhambits)]]:
        ancilla = qarray(n)
        ancilla_index = 0
        for op in ops:
            if op[2] == -1:
                _half_adder_with_swaps(main_reg, ancilla, op, ancilla_index)
            else:
                _full_adder_with_swaps(main_reg, ancilla, op, ancilla_index)
            ancilla_index += 1

        # now the hamming weight bits will be at idxs, so we need to do some more swaps
        # swap first into ancilla to prevent any conflicts
        for k in range(len(idxs)):
            mem_swap(main_reg[idxs[k]], ancilla[k])
        # then swap into the first nhambits of the main reg
        for i in range(nhambits):
            mem_swap(main_reg[i], ancilla[i])

        # split the arrays to the correct sizes (cursed)
        tempham = array(nothing[qubit]() for _ in range(nhambits))
        tempjunk = array(nothing[qubit]() for _ in range(comptime(2 * n - nhambits)))

        mainopt = array(nothing[qubit]() for _ in range(n))
        for i in range(len(main_reg)):
            mainopt[i].swap(some(main_reg.take(i))).unwrap_nothing()

        for i in range(n):
            if i < nhambits:
                mem_swap(tempham[i], mainopt[i])
            else:
                mem_swap(tempjunk[i - nhambits], mainopt[i])

        ancillaopt = array(some(ancilla_q) for ancilla_q in ancilla)
        for i in range(n):
            mem_swap(tempjunk[i + comptime(n - nhambits)], ancillaopt[i])

        for opt in mainopt:
            opt.unwrap_nothing()
        for opt in ancillaopt:
            opt.unwrap_nothing()

        ham_reg = array(t.unwrap() for t in tempham)
        junk_reg = array(t.unwrap() for t in tempjunk)
        return ham_reg, junk_reg

    return hamm


def hamming_weight_func_inv[n_q: nat, logn_q: nat, nmlogn_q: nat](
    n: int,
) -> GuppyFunctionDefinition[
    [array[qubit, n_q], array[qubit, logn_q], array[qubit, nmlogn_q]],
    None,
]:
    """Produce a guppy function for uncomputing the hamming weight of a register.

    The circuit takes in nhambits qubits in `ham_reg` and 2*n-nhambits junk qubits
    in `junk_reg`, as produced by `hamming_weight_func` and uncomputes the the circuit
    produced by that function.

    UNSAFE:
        Must take in main_reg as the first argument, which is an empty array of size n
        where all qubits have been removed using `.take`
    """
    nhambits = num_hamming_weight_bits(n)
    ops, idxs = _adder_indices(n)

    @guppy
    @no_type_check
    def hamm_uncompute(
        main_reg: array[qubit, n],
        ham_reg: array[qubit, nhambits] @ owned,
        junk_reg: array[qubit, comptime(2 * n - nhambits)] @ owned,
    ) -> None:
        # join the arrays to the correct sizes
        temporig = array(nothing[qubit]() for _ in range(n))
        tempancilla = array(nothing[qubit]() for _ in range(n))
        hamopt = array(some(ham_q) for ham_q in ham_reg)
        junkopt = array(some(junk_q) for junk_q in junk_reg)

        for i in range(n):
            if i < nhambits:
                mem_swap(temporig[i], hamopt[i])
            else:
                mem_swap(temporig[i], junkopt[i - nhambits])

        for i in range(n):
            mem_swap(tempancilla[i], junkopt[i + (comptime(n - nhambits))])

        for opt in hamopt:
            opt.unwrap_nothing()
        for opt in junkopt:
            opt.unwrap_nothing()

        main_temp = array(t.unwrap() for t in temporig)
        for i in range(n):
            main_reg.put(main_temp.take(i), i)
        main_temp.discard_all_taken()
        ancilla_reg = array(t.unwrap() for t in tempancilla)

        # move hamming bits back to the indices from the end of the comp step
        for i in range(nhambits):
            mem_swap(main_reg[i], ancilla_reg[i])

        for k in range(len(idxs)):
            mem_swap(main_reg[idxs[k]], ancilla_reg[k])

        # undo adders
        ancilla_index = len(ops) - 1
        for op in comptime(ops[::-1]):
            if op[2] == -1:
                _half_adder_with_swaps_inv(main_reg, ancilla_reg, op, ancilla_index)
            else:
                _full_adder_with_swaps_inv(main_reg, ancilla_reg, op, ancilla_index)
            ancilla_index -= 1

        discard_array(ancilla_reg)

    return hamm_uncompute
