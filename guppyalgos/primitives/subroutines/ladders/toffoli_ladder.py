"""Implementations for CToffoli ladders."""

from math import floor, log2
from typing import no_type_check

from guppylang import guppy
from guppylang.std.array import frozenarray
from guppylang.std.builtins import array, comptime, control, dagger, nat
from guppylang.std.quantum import discard, qubit, toffoli

from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.primitives.gate_decompositions.cnx.cnx import CNXPartialAndLadder
from guppyalgos.primitives.measurement.utils import discard_array
from guppyalgos.primitives.subroutines.ladders.ladder import LadderIndexing
from guppyalgos.utils import qarray

LabeledIndex = tuple[int, int]
IndexTriple = tuple[int, int, int]
LabeledIndexTriple = tuple[LabeledIndex, LabeledIndex, LabeledIndex]


@guppy.struct
class ToffoliLadderLinear:
    """Linear-depth Toffoli ladder.

    Protocols: Ladder
    """

    @guppy(unitary=True)
    @no_type_check
    def ascending[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply ascending linear Toffoli ladder."""
        _toffoli_ladder_apply_from_inds(
            qs,
            comptime(_lin_toffoli_ladder_indices(n)),
        )

    @guppy(unitary=True)
    @no_type_check
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

    @guppy.unitary
    class ascending[n_q: nat]:  # noqa: D106
        @guppy
        @no_type_check
        def __call__(self, qs: array[qubit, n_q]) -> None:
            """Apply ascending log depth Toffoli ladder."""
            anc = qarray(comptime(log_toffoli_ladder_num_ancilla(n_q)))
            self.ascending_with_cca(qs, anc)
            discard_array(anc)

        @guppy
        @no_type_check
        def controlled[n_ctrls: nat](  # noqa: D102
            self, qs: array[qubit, n_q], controls: array[qubit, n_ctrls]
        ) -> None:
            anc = qarray(comptime(log_toffoli_ladder_num_ancilla(n_q)))
            with control(controls):
                self.ascending_with_cca(qs, anc)
            discard_array(anc)

        @guppy
        @no_type_check
        def daggered(self, qs: array[qubit, n_q]) -> None:  # noqa: D102
            anc = qarray(comptime(log_toffoli_ladder_num_ancilla(n_q)))
            with dagger:
                self.ascending_with_cca(qs, anc)
            discard_array(anc)

        @guppy
        @no_type_check
        def ctrl_daggered[n_ctrls: nat](  # noqa: D102
            self, qs: array[qubit, n_q], controls: array[qubit, n_ctrls]
        ) -> None:
            anc = qarray(comptime(log_toffoli_ladder_num_ancilla(n_q)))
            with control(controls), dagger:
                self.ascending_with_cca(qs, anc)
            discard_array(anc)

    @guppy.unitary
    class descending[n_q: nat]:  # noqa: D106
        @guppy
        @no_type_check
        def __call__(self, qs: array[qubit, n_q]) -> None:
            """Apply descending log depth Toffoli ladder."""
            anc = qarray(comptime(log_toffoli_ladder_num_ancilla(n_q)))
            self.descending_with_cca(qs, anc)
            discard_array(anc)

        @guppy
        @no_type_check
        def controlled[n_ctrls: nat](  # noqa: D102
            self, qs: array[qubit, n_q], controls: array[qubit, n_ctrls]
        ) -> None:
            anc = qarray(comptime(log_toffoli_ladder_num_ancilla(n_q)))
            with control(controls):
                self.descending_with_cca(qs, anc)
            discard_array(anc)

        @guppy
        @no_type_check
        def daggered(self, qs: array[qubit, n_q]) -> None:  # noqa: D102
            anc = qarray(comptime(log_toffoli_ladder_num_ancilla(n_q)))
            with dagger:
                self.descending_with_cca(qs, anc)
            discard_array(anc)

        @guppy
        @no_type_check
        def ctrl_daggered[n_ctrls: nat](  # noqa: D102
            self, qs: array[qubit, n_q], controls: array[qubit, n_ctrls]
        ) -> None:
            anc = qarray(comptime(log_toffoli_ladder_num_ancilla(n_q)))
            with control(controls), dagger:
                self.descending_with_cca(qs, anc)
            discard_array(anc)

    @guppy(unitary=True)
    @no_type_check
    def ascending_with_cca[n_q: nat, n_a: nat](
        self, qs: array[qubit, n_q], anc: array[qubit, n_a]
    ) -> None:
        """Apply ascending log depth Toffoli ladder on the given ancillae."""
        _toffoli_ladder_apply_from_labeled_inds(
            qs,
            comptime(_log_toffoli_ladder_labeled_indices(n_q)),
            anc,
        )

    @guppy(unitary=True)
    @no_type_check
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
    def n_gates(self, n_qubits: int) -> int:
        """Return the number of Toffoli gates in the ladder."""
        return len(_log_toffoli_ladder_labeled_indices(n_qubits))

    @guppy
    def num_ancilla(self, n_q: int) -> int:
        """Return the number of ancilla qubits."""
        return log_toffoli_ladder_num_ancilla(n_q)


# Compiled wrappers, as `with dagger` is not supported directly in `guppy.comptime`;
# see https://github.com/Quantinuum/guppylang/issues/1701.
@guppy(unitary=True)
@no_type_check
def log_ladder_ascending_dagger[n: nat](qs: array[qubit, n]) -> None:
    """Apply the daggered ascending log depth Toffoli ladder."""
    with dagger:
        ToffoliLadderLog().ascending(qs)


@guppy(unitary=True)
@no_type_check
def log_ladder_ascending_dagger_with_cca[n: nat, n_a: nat](
    qs: array[qubit, n], anc: array[qubit, n_a]
) -> None:
    """Apply the daggered ascending log depth Toffoli ladder on the given ancillae."""
    with dagger:
        ToffoliLadderLog().ascending_with_cca(qs, anc)


def _reflect_ladder(
    asc_inds: list[IndexTriple],
) -> list[IndexTriple]:
    n_qubits = 1 + max((index for gate in asc_inds for index in gate), default=0)
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
        (index for gate in asc_inds for register, index in gate if register == 0),
        default=0,
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
def _toffoli_extra_ctrl[n_q: nat](
    q: array[qubit, n_q], i: int, j: int, k: int, ctrl: qubit
) -> None:
    """Apply a Toffoli with one extra control via a temporary AND."""
    t = qubit()
    temp_and_compute(q[i], q[j], t)
    toffoli(ctrl, t, q[k])
    temp_and_uncompute(q[i], q[j], t)
    discard(t)


@guppy.unitary
class _toffoli_ladder_apply_from_inds[n_q: nat, n_gates: nat]:
    @guppy
    @no_type_check
    def __call__(
        q: array[qubit, n_q],
        gate_indices: frozenarray[tuple[int, int, int], n_gates],
    ) -> None:
        """Apply Toffoli gates based on the provided indices."""
        for i, j, k in gate_indices:
            if i < n_q and j < n_q and k < n_q:
                toffoli(q[i], q[j], q[k])

    @guppy
    @no_type_check
    def daggered(
        q: array[qubit, n_q],
        gate_indices: frozenarray[tuple[int, int, int], n_gates],
    ) -> None:
        """Apply the self-inverse Toffoli gates in reverse order."""
        reversed_inds = gate_indices.mutable_copy()
        reversed_inds.reverse_in_place()
        for i, j, k in reversed_inds:
            if i < n_q and j < n_q and k < n_q:
                toffoli(q[i], q[j], q[k])

    @guppy
    @no_type_check
    def controlled[n_controls: nat](
        q: array[qubit, n_q],
        gate_indices: frozenarray[tuple[int, int, int], n_gates],
        controls: array[qubit, n_controls],
    ) -> None:
        """Apply the Toffoli gates under the AND of the controls."""
        ctrl = qubit()
        work = CNXPartialAndLadder.compute(controls, ctrl)
        for i, j, k in gate_indices:
            if i < n_q and j < n_q and k < n_q:
                _toffoli_extra_ctrl(q, i, j, k, ctrl)
        CNXPartialAndLadder.uncompute(controls, work, ctrl)
        discard(ctrl)

    @guppy
    @no_type_check
    def ctrl_daggered[n_controls: nat](
        q: array[qubit, n_q],
        gate_indices: frozenarray[tuple[int, int, int], n_gates],
        controls: array[qubit, n_controls],
    ) -> None:
        """Apply the controlled, self-inverse Toffoli gates in reverse order."""
        reversed_inds = gate_indices.mutable_copy()
        reversed_inds.reverse_in_place()
        ctrl = qubit()
        work = CNXPartialAndLadder.compute(controls, ctrl)
        for i, j, k in reversed_inds:
            if i < n_q and j < n_q and k < n_q:
                _toffoli_extra_ctrl(q, i, j, k, ctrl)
        CNXPartialAndLadder.uncompute(controls, work, ctrl)
        discard(ctrl)


@guppy
@no_type_check
def _labeled_toffoli[n_q: nat, n_a: nat](
    q: array[qubit, n_q],
    anc: array[qubit, n_a],
    gate: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> None:
    """Apply a Toffoli to the qubits the labeled triple addresses.

    Each operand is a ``(register, index)`` pair: register 0 is the data
    register ``q`` and register 1 is the ancilla register ``anc``.
    """
    q_i, q_j, q_k = gate
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


@guppy
@no_type_check
def _labeled_toffoli_extra_ctrl[n_q: nat, n_a: nat](
    q: array[qubit, n_q],
    anc: array[qubit, n_a],
    gate: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    ctrl: qubit,
) -> None:
    """Apply the labeled Toffoli with one extra control via a temporary AND."""
    q_i, q_j, q_k = gate
    ri, i = q_i
    rj, j = q_j
    rk, k = q_k

    t = qubit()
    if ri == 0 and rj == 0:
        temp_and_compute(q[i], q[j], t)
    elif ri == 0:
        temp_and_compute(q[i], anc[j], t)
    elif rj == 0:
        temp_and_compute(anc[i], q[j], t)
    else:
        temp_and_compute(anc[i], anc[j], t)

    if rk == 0:
        toffoli(ctrl, t, q[k])
    else:
        toffoli(ctrl, t, anc[k])

    if ri == 0 and rj == 0:
        temp_and_uncompute(q[i], q[j], t)
    elif ri == 0:
        temp_and_uncompute(q[i], anc[j], t)
    elif rj == 0:
        temp_and_uncompute(anc[i], q[j], t)
    else:
        temp_and_uncompute(anc[i], anc[j], t)
    discard(t)


@guppy.unitary
class _toffoli_ladder_apply_from_labeled_inds[n_q: nat, n_gates: nat, n_a: nat]:
    @guppy
    @no_type_check
    def __call__(
        q: array[qubit, n_q],
        labeled_indices: frozenarray[
            tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
            n_gates,
        ],
        anc: array[qubit, n_a],
    ) -> None:
        """Apply Toffoli gates based on the provided indices."""
        for gate in labeled_indices:
            _labeled_toffoli(q, anc, gate)

    @guppy
    @no_type_check
    def daggered(
        q: array[qubit, n_q],
        labeled_indices: frozenarray[
            tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
            n_gates,
        ],
        anc: array[qubit, n_a],
    ) -> None:
        """Apply the self-inverse Toffoli gates in reverse order."""
        reversed_inds = labeled_indices.mutable_copy()
        reversed_inds.reverse_in_place()
        for gate in reversed_inds:
            _labeled_toffoli(q, anc, gate)

    @guppy
    @no_type_check
    def controlled[n_controls: nat](
        q: array[qubit, n_q],
        labeled_indices: frozenarray[
            tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
            n_gates,
        ],
        anc: array[qubit, n_a],
        controls: array[qubit, n_controls],
    ) -> None:
        """Apply the Toffoli gates under the AND of the controls."""
        ctrl = qubit()
        work = CNXPartialAndLadder.compute(controls, ctrl)
        for gate in labeled_indices:
            _labeled_toffoli_extra_ctrl(q, anc, gate, ctrl)
        CNXPartialAndLadder.uncompute(controls, work, ctrl)
        discard(ctrl)

    @guppy
    @no_type_check
    def ctrl_daggered[n_controls: nat](
        q: array[qubit, n_q],
        labeled_indices: frozenarray[
            tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
            n_gates,
        ],
        anc: array[qubit, n_a],
        controls: array[qubit, n_controls],
    ) -> None:
        """Apply the controlled, self-inverse Toffoli gates in reverse order."""
        reversed_inds = labeled_indices.mutable_copy()
        reversed_inds.reverse_in_place()
        ctrl = qubit()
        work = CNXPartialAndLadder.compute(controls, ctrl)
        for gate in reversed_inds:
            _labeled_toffoli_extra_ctrl(q, anc, gate, ctrl)
        CNXPartialAndLadder.uncompute(controls, work, ctrl)
        discard(ctrl)


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
            return n - i - 2 * (n // 2**i) - m.bit_count()

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
