"""Implementations for CX ladders."""

from guppyalgos.primitives.subroutines.ladders.ladder import LadderIndexing

from typing import cast, no_type_check

from guppylang import comptime, guppy
from guppylang.std.array import frozenarray
from guppylang.std.builtins import array, nat, control
from guppylang.std.quantum import cx, qubit


@guppy.struct
class CXLadderLinear:
    """Linear depth CX Ladder.

    Protocols: Ladder
    """

    @guppy
    @no_type_check
    def ascending[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply ascending linear CX ladder."""
        _cx_ladder_apply_from_inds(
            qs,
            comptime(_lin_cx_ladder_indices(n)),
        )

    @guppy
    @no_type_check
    def ascending_dagger[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply ascending linear CX ladder dagger."""
        _cx_ladder_apply_from_inds(
            qs,
            comptime(
                ladder_inds_from_ascending(
                    _lin_cx_ladder_indices(n), LadderIndexing.ASCENDING_DAGGER
                )
            ),
        )

    @guppy
    @no_type_check
    def descending[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply descending linear CX ladder."""
        _cx_ladder_apply_from_inds(
            qs,
            comptime(
                ladder_inds_from_ascending(
                    _lin_cx_ladder_indices(n), LadderIndexing.DESCENDING
                )
            ),
        )

    @guppy
    @no_type_check
    def descending_dagger[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply descending linear CX ladder dagger."""
        _cx_ladder_apply_from_inds(
            qs,
            comptime(
                ladder_inds_from_ascending(
                    _lin_cx_ladder_indices(n), LadderIndexing.DESCENDING_DAGGER
                )
            ),
        )


@guppy.struct
class CXLadderLog:
    """Log depth CX Ladder from https://arxiv.org/abs/2501.16802.

    Log-depth increases the gate count.

    Protocols: Ladder
    """

    @guppy
    @no_type_check
    def ascending[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply ascending log-depth CX ladder."""
        _cx_ladder_apply_from_inds(
            qs,
            comptime(log_cx_ladder_indices(n)),
        )

    @guppy
    @no_type_check
    def ascending_dagger[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply ascending log-depth CX ladder dagger."""
        _cx_ladder_apply_from_inds(
            qs,
            comptime(
                ladder_inds_from_ascending(
                    log_cx_ladder_indices(n), LadderIndexing.ASCENDING_DAGGER
                )
            ),
        )

    @guppy
    @no_type_check
    def descending[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply descending log-depth CX ladder."""
        _cx_ladder_apply_from_inds(
            qs,
            comptime(
                ladder_inds_from_ascending(
                    log_cx_ladder_indices(n), LadderIndexing.DESCENDING
                )
            ),
        )

    @guppy
    @no_type_check
    def descending_dagger[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply descending log-depth CX ladder dagger."""
        _cx_ladder_apply_from_inds(
            qs,
            comptime(
                ladder_inds_from_ascending(
                    log_cx_ladder_indices(n), LadderIndexing.DESCENDING_DAGGER
                )
            ),
        )


@guppy.unitary
class _cx_ladder_apply_from_inds:
    @guppy
    @no_type_check
    def __call__[n_qubits: nat, n_gates: nat](
        q: array[qubit, n_qubits],
        gate_indices: frozenarray[tuple[int, int], n_gates],
    ) -> None:
        """Apply CX gates based on the provided indices."""
        for i, j in gate_indices:
            cx(q[i], q[j])

    @guppy
    @no_type_check
    def daggered[n_qubits: nat, n_gates: nat](
        q: array[qubit, n_qubits],
        gate_indices: frozenarray[tuple[int, int], n_gates],
    ) -> None:
        """Apply CX gates based on the provided indices."""
        reversed_inds = gate_indices.mutable_copy()
        reversed_inds.reverse_in_place()
        for i, j in reversed_inds:
            cx(q[i], q[j])

    @guppy
    @no_type_check
    def controlled[n_qubits: nat, n_gates: nat, n_ctrls: nat](
        q: array[qubit, n_qubits],
        gate_indices: frozenarray[tuple[int, int], n_gates],
        ctrls: array[qubit, n_ctrls],
    ) -> None:
        """Apply CX gates based on the provided indices."""
        for i, j in gate_indices:
            with control(ctrls):
                cx(q[i], q[j])

    @guppy
    @no_type_check
    def ctrl_daggered[n_qubits: nat, n_gates: nat, n_ctrls: nat](
        q: array[qubit, n_qubits],
        gate_indices: frozenarray[tuple[int, int], n_gates],
        ctrls: array[qubit, n_ctrls],
    ) -> None:
        """Apply CX gates based on the provided indices."""
        reversed_inds = gate_indices.mutable_copy()
        reversed_inds.reverse_in_place()
        for i, j in reversed_inds:
            with control(ctrls):
                cx(q[i], q[j])


def ladder_inds_from_ascending(
    asc_inds: list[tuple[int, int]], index_type: LadderIndexing
) -> list[tuple[int, int]]:
    """Derive the gate indices of a ladder variant from its ascending indices.

    Index generation is separated into a method for testing.
    """
    match index_type:
        case LadderIndexing.ASCENDING:
            return asc_inds
        case LadderIndexing.ASCENDING_DAGGER:
            return asc_inds[::-1]
        case LadderIndexing.DESCENDING_DAGGER:
            return [(j, i) for (i, j) in asc_inds]
        case LadderIndexing.DESCENDING:
            return [(j, i) for (i, j) in asc_inds][::-1]


def _lin_cx_ladder_indices(n_qubits: int) -> list[tuple[int, int]]:
    """Generate pairs of indices for ascending linear ladder.

    Index generation is separated into a method for testing.
    """
    return [(i, i + 1) for i in range(n_qubits - 1)]


def log_cx_ladder_indices(n_qubits: int) -> list[tuple[int, int]]:
    """Generate indices of control and target qubits for ascending log-depth CX ladder.

    The algorithm is provided in the following paper (Algorithm 1, Section 3):
    https://arxiv.org/abs/2501.16802
    The actual implementation of this algorithm is the inner function.
    The reason behind this nested function is that the algorithm
    works with a list of indices (by recursively slicing them), but we use
    n_qubits as an input in our ladder implementations.

    Args:
        n_qubits (int): The number of qubits.

    Returns:
        list[tuple[int, int]]: List of tuples representing pairs of indices.

    """

    def log_cx_ladder_indices(list_of_indices: list[int]) -> list[tuple[int, int]]:
        n = len(list_of_indices)
        if n == 1:
            return []
        elif n == 2:
            return [cast(tuple[int, int], tuple(list_of_indices))]
        next_index_batch = [list_of_indices[1]]
        c_r_indices = [(list_of_indices[0], list_of_indices[1])]
        c_l_indices = [(list_of_indices[n - 2], list_of_indices[n - 1])]
        midpoint = n / 2
        for i in range(1, int(midpoint) + (midpoint > int(midpoint)) - 1):
            c_r_indices.append((list_of_indices[2 * i], list_of_indices[2 * i + 1]))
            c_l_indices.append((list_of_indices[2 * i - 1], list_of_indices[2 * i]))
            next_index_batch.append(list_of_indices[2 * i + 1])
        if n % 2 == 0:
            next_index_batch.append(list_of_indices[n - 2])
        res = [
            *c_l_indices,
            *log_cx_ladder_indices(next_index_batch),
            *reversed(c_r_indices),
        ]
        return res

    inds = log_cx_ladder_indices(list(range(n_qubits)))
    assert all(
        (i >= 0 and i < n_qubits) and (j >= 0 and j < n_qubits) for (i, j) in inds
    )
    return inds[::-1]


def _ghz_state_prep_filter(tuples: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Eliminate redundant gates in GHZ state prep using ascending log ladder."""
    if len(tuples) == 0:
        raise ValueError("GHZ state prep must be called on an array of size >1")
    result = [tuples[0]]
    seen_values = {
        tuples[0][0],
        tuples[0][1],
    }

    for t in tuples[1:]:
        if (t[0] in seen_values) or (t[1] in seen_values):
            result.append(t)
            seen_values.add(t[0])
            seen_values.add(t[1])

    return result
