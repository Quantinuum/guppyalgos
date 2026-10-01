"""Implementations for CToffoli ladders."""

from math import floor, log2
from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.array import frozenarray
from guppylang.std.quantum import qubit, toffoli
from guppyalgos.utils import qarray
from guppyalgos.primitives.measurement.utils import discard_array

from guppyalgos.primitives.subroutines.ladders.ladder import LadderIndexing


LabeledIndex = tuple[int, int]
IndexTriple = tuple[int, int, int]
LabeledIndexTriple = tuple[LabeledIndex, LabeledIndex, LabeledIndex]


@guppy.struct
class ToffoliLadderLinear:
    """Linear-depth Toffoli ladder.

    Protocols: Ladder
    """

    @guppy
    def ascending[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply ascending linear Toffoli ladder."""
        _toffoli_ladder_apply_from_inds(
            qs,
            comptime(_lin_toffoli_ladder_indices(n)),
        )

    @guppy
    def ascending_dagger[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply ascending linear Toffoli ladder dagger."""
        _toffoli_ladder_apply_from_inds(
            qs,
            comptime(
                _ladder_inds_from_ascending(
                    _lin_toffoli_ladder_indices(n), LadderIndexing.ASCENDING_DAGGER
                )
            ),
        )

    @guppy
    def descending[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply descending linear Toffoli ladder."""
        _toffoli_ladder_apply_from_inds(
            qs,
            comptime(
                _ladder_inds_from_ascending(
                    _lin_toffoli_ladder_indices(n), LadderIndexing.DESCENDING
                )
            ),
        )

    @guppy
    def descending_dagger[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply descending linear Toffoli ladder dagger."""
        _toffoli_ladder_apply_from_inds(
            qs,
            comptime(
                _ladder_inds_from_ascending(
                    _lin_toffoli_ladder_indices(n), LadderIndexing.DESCENDING_DAGGER
                )
            ),
        )

    @guppy
    def n_gates(self, n_qubits: int) -> int:
        """Return the number of Toffoli gates in the ladder."""
        return (n_qubits - 1) // 2

    @guppy
    def num_ancilla(self, n: int) -> int:
        """Return the number of ancillas required for this ladder."""
        return 0


@guppy.struct
class ToffoliLadderLog:
    """Log depth Toffoli ladder from https://arxiv.org/abs/2510.00840 (Algorithm 1).

    Log depth increases the gate count.

    Protocols: Ladder
    """

    @guppy
    def ascending[n_q: nat](self, qs: array[qubit, n_q]) -> None:
        """Apply ascending log depth Toffoli ladder."""
        anc = qarray(comptime(log_toffoli_ladder_num_ancilla(n_q)))
        self.ascending_with_cca(qs, anc)  # ty: ignore[missing-argument]
        discard_array(anc)

    @guppy
    def ascending_dagger[n_q: nat](self, qs: array[qubit, n_q]) -> None:
        """Apply ascending log depth Toffoli ladder dagger."""
        anc = qarray(comptime(log_toffoli_ladder_num_ancilla(n_q)))
        self.ascending_dagger_with_cca(qs, anc)  # ty: ignore[missing-argument]
        discard_array(anc)

    @guppy
    def descending[n_q: nat](self, qs: array[qubit, n_q]) -> None:
        """Apply descending log depth Toffoli ladder."""
        anc = qarray(comptime(log_toffoli_ladder_num_ancilla(n_q)))
        self.descending_with_cca(qs, anc)  # ty: ignore[missing-argument]
        discard_array(anc)

    @guppy
    def descending_dagger[n_q: nat](self, qs: array[qubit, n_q]) -> None:
        """Apply descending log depth Toffoli ladder dagger."""
        anc = qarray(comptime(log_toffoli_ladder_num_ancilla(n_q)))
        self.descending_dagger_with_cca(qs, anc)  # ty: ignore[missing-argument]
        discard_array(anc)

    @guppy
    def ascending_with_cca[n_q: nat, n_a: nat](
        self, qs: array[qubit, n_q], anc: array[qubit, n_a]
    ) -> None:
        """Apply ascending log depth Toffoli ladder on the given ancillae."""
        _toffoli_ladder_apply_from_labeled_inds(
            qs,
            comptime(_log_toffoli_ladder_labeled_indices(n_q)),
            anc,
        )

    @guppy
    def ascending_dagger_with_cca[n_q: nat, n_a: nat](
        self, qs: array[qubit, n_q], anc: array[qubit, n_a]
    ) -> None:
        """Apply ascending log depth Toffoli ladder dagger on the given ancillae."""
        _toffoli_ladder_apply_from_labeled_inds(
            qs,
            comptime(
                _ladder_labeled_inds_from_ascending(
                    _log_toffoli_ladder_labeled_indices(n_q),
                    LadderIndexing.ASCENDING_DAGGER,
                )
            ),
            anc,
        )

    @guppy
    def descending_with_cca[n_q: nat, n_a: nat](
        self, qs: array[qubit, n_q], anc: array[qubit, n_a]
    ) -> None:
        """Apply descending log depth Toffoli ladder on the given ancillae."""
        _toffoli_ladder_apply_from_labeled_inds(
            qs,
            comptime(
                _ladder_labeled_inds_from_ascending(
                    _log_toffoli_ladder_labeled_indices(n_q), LadderIndexing.DESCENDING
                )
            ),
            anc,
        )

    @guppy
    def descending_dagger_with_cca[n_q: nat, n_a: nat](
        self, qs: array[qubit, n_q], anc: array[qubit, n_a]
    ) -> None:
        """Apply descending log depth Toffoli ladder dagger on the given ancillae."""
        _toffoli_ladder_apply_from_labeled_inds(
            qs,
            comptime(
                _ladder_labeled_inds_from_ascending(
                    _log_toffoli_ladder_labeled_indices(n_q),
                    LadderIndexing.DESCENDING_DAGGER,
                )
            ),
            anc,
        )

    @guppy
    def n_gates(self, n_qubits: int) -> int:
        """Return the number of Toffoli gates in the ladder."""
        return len(_log_toffoli_ladder_labeled_indices(n_qubits))

    @guppy
    def num_ancilla(self, n_q: int) -> int:
        """Return the number of ancilla qubits."""
        return log_toffoli_ladder_num_ancilla(n_q)


def _reflect_ladder(
    asc_inds: list[IndexTriple],
) -> list[IndexTriple]:
    n_qubits = 1 + max(index for gate in asc_inds for index in gate)
    return [
        (
            n_qubits - 1 - i,
            n_qubits - 1 - j,
            n_qubits - 1 - k,
        )
        for i, j, k in asc_inds
    ]


def _reflect_labeled_index(
    labeled_index: tuple[int, int],
    n_qubits: int,
) -> tuple[int, int]:
    register, index = labeled_index

    if register == 0:
        return register, n_qubits - 1 - index

    return register, index


def _reflect_labeled_ladder(
    asc_inds: list[LabeledIndexTriple],
) -> list[LabeledIndexTriple]:
    n_qubits = 1 + max(
        index for gate in asc_inds for register, index in gate if register == 0
    )
    return [
        (
            _reflect_labeled_index(i, n_qubits),
            _reflect_labeled_index(j, n_qubits),
            _reflect_labeled_index(k, n_qubits),
        )
        for i, j, k in asc_inds
    ]


@guppy
@no_type_check
def _toffoli_ladder_apply_from_inds[n_q: nat, n_gates: nat](
    q: array[qubit, n_q],
    gate_indices: frozenarray[tuple[int, int, int], n_gates],
) -> None:
    """Apply Toffoli gates based on the provided indices."""
    for i, j, k in gate_indices:
        if i < n_q and j < n_q and k < n_q:
            toffoli(q[i], q[j], q[k])


@guppy
@no_type_check
def _toffoli_ladder_apply_from_labeled_inds[n_q: nat, n_gates: nat, n_a: nat](
    q: array[qubit, n_q],
    labeled_indices: frozenarray[
        tuple[
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
        ],
        n_gates,
    ],
    anc: array[qubit, n_a],
) -> None:
    """Apply Toffoli gates based on the provided indices."""
    for q_i, q_j, q_k in labeled_indices:
        ri, i = q_i
        rj, j = q_j
        rk, k = q_k

        if ri == 0:
            if rj == 0:
                if rk == 0:
                    toffoli(q[i], q[j], q[k])
                else:
                    toffoli(q[i], q[j], anc[k])
            else:
                if rk == 0:
                    toffoli(q[i], anc[j], q[k])
                else:
                    toffoli(q[i], anc[j], anc[k])
        else:
            if rj == 0:
                if rk == 0:
                    toffoli(anc[i], q[j], q[k])
                else:
                    toffoli(anc[i], q[j], anc[k])
            else:
                if rk == 0:
                    toffoli(anc[i], anc[j], q[k])
                else:
                    toffoli(anc[i], anc[j], anc[k])


def _ladder_inds_from_ascending(
    asc_inds: list[IndexTriple], index_type: LadderIndexing
) -> list[IndexTriple]:
    match index_type:
        case LadderIndexing.ASCENDING:
            return asc_inds
        case LadderIndexing.ASCENDING_DAGGER:
            return asc_inds[::-1]
        case LadderIndexing.DESCENDING_DAGGER:
            return _reflect_ladder(asc_inds)[::-1]
        case LadderIndexing.DESCENDING:
            return _reflect_ladder(asc_inds)


def _ladder_labeled_inds_from_ascending(
    asc_inds: list[LabeledIndexTriple], index_type: LadderIndexing
) -> list[IndexTriple]:
    match index_type:
        case LadderIndexing.ASCENDING:
            return asc_inds
        case LadderIndexing.ASCENDING_DAGGER:
            return asc_inds[::-1]
        case LadderIndexing.DESCENDING_DAGGER:
            return _reflect_labeled_ladder(asc_inds)[::-1]
        case LadderIndexing.DESCENDING:
            return _reflect_labeled_ladder(asc_inds)


def _lin_toffoli_ladder_indices(n_qubits: int) -> list[IndexTriple]:
    """Generate pairs of indices for ascending linear ladder.

    Index generation is separated into a method for testing.
    """
    assert n_qubits % 2, "Toffoli ladder acts on odd number of qubits."
    return [(i, i + 1, i + 2) for i in range(0, n_qubits - 1, 2)]


def log_toffoli_ladder_num_ancilla(n_qubits: int) -> int:
    """Count the clean ancillae the log depth Toffoli ladder needs.

    Args:
        n_qubits: Number of data qubits the ladder acts on.

    """
    if n_qubits <= 3:
        return 0

    n = (n_qubits + 1) // 2
    return n - floor(log2(n)) - n.bit_count()


def _log_toffoli_ladder_labeled_indices(
    n_qubits: int,
) -> list[LabeledIndexTriple]:
    """Generate indices for the log depth CToffoli ladder.

    The algorithm is provided in Algorithm 1, Section 3 of
    https://arxiv.org/abs/2510.00840.

    Args:
        n_qubits: Number of data qubits.

    Returns:
        Triples of ``(register, index)`` references. ``q`` denotes the data
        register and ``a`` denotes the ancilla register.

    """
    assert n_qubits % 2, "Toffoli ladder acts on odd number of qubits."

    if n_qubits == 1:
        return []
    if n_qubits == 3:
        return [((0, 0), (0, 1), (0, 2))]

    n = (n_qubits + 1) // 2
    n_a = log_toffoli_ladder_num_ancilla(n_qubits)

    def log_toffoli_ladder_labeled_indices(
        list_of_indices: list[int],
    ) -> list[LabeledIndexTriple]:
        n_q = len(list_of_indices)

        def sigma(i: int) -> int:
            m = n & ((1 << i) - 1)
            return n - i - 2 * n // 2**i - m.bit_count()

        a_indices = [(0, i) for i in range(0, n_q, 2)]
        b_indices = [(0, i) for i in range(1, n_q, 2)]
        c_indices = [(1, i) for i in range(n_a)]

        ccx_indices: list[LabeledIndexTriple] = []

        # Slice 1.
        for j in range(1, n // 2):
            ccx_indices.append(
                (b_indices[2 * j - 1], b_indices[2 * j], c_indices[j - 1])
            )

        for i in range(2, floor(log2(n))):
            for j in range(1, n // (2**i)):
                ccx_indices.append(
                    (
                        c_indices[2 * j + sigma(i - 1)],
                        c_indices[2 * j + 1 + sigma(i - 1)],
                        c_indices[j + sigma(i)],
                    )
                )

        # Slice 2.
        for j in range(1, (n - 1) // 2 + 1):
            ccx_indices.append(
                (a_indices[2 * j - 1], b_indices[2 * j - 1], a_indices[2 * j])
            )

        for i in range(2, floor(log2(2 * n / 3)) + 1):
            for j in range(1, (n - 2 ** (i - 1)) // 2**i + 1):
                ccx_indices.append(
                    (
                        a_indices[2**i * j - 1],
                        c_indices[2 * j + sigma(i - 1)],
                        a_indices[2**i * j + 2 ** (i - 1) - 1],
                    )
                )

        # Slice 3.
        for i in range(floor(log2(n)), 1, -1):
            for j in range(1, floor(n // 2**i) + 1):
                ccx_indices.append(
                    (
                        a_indices[2**i * j - 2 ** (i - 1) - 1],
                        c_indices[2 * j - 1 + sigma(i - 1)],
                        a_indices[2**i * j - 1],
                    )
                )

        for j in range(1, n // 2 + 1):
            ccx_indices.append(
                (a_indices[2 * j - 2], b_indices[2 * j - 2], a_indices[2 * j - 1])
            )

        # Uncompute slice 1.
        for i in range(floor(log2(n)) - 1, 1, -1):
            for j in range(n // (2**i) - 1, 0, -1):
                ccx_indices.append(
                    (
                        c_indices[2 * j + sigma(i - 1)],
                        c_indices[2 * j + 1 + sigma(i - 1)],
                        c_indices[j + sigma(i)],
                    )
                )

        for j in range(n // 2 - 1, 0, -1):
            ccx_indices.append(
                (b_indices[2 * j - 1], b_indices[2 * j], c_indices[j - 1])
            )

        return ccx_indices

    labeled_inds = log_toffoli_ladder_labeled_indices(list(range(n_qubits)))
    assert all(
        (reg == 0 and 0 <= i < n_qubits) or (reg == 1 and 0 <= i < n_a)
        for gate in labeled_inds
        for reg, i in gate
    )

    return labeled_inds[::-1]
