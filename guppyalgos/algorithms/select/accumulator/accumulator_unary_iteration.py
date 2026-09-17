"""Meta function to create an accumulator-based unary iteration Select function."""

from guppylang import guppy
from guppylang import comptime

from guppylang.std.quantum import cx, qubit, reset, x
from guppylang.std.builtins import array, frozenarray, nat

from collections.abc import Callable
from typing import no_type_check

from guppyalgos.primitives.gate_decompositions.and_op import (
    index_and,
    temp_and_comp_index,
    temp_and_uncomp_index,
)
from guppyalgos.primitives.measurement import discard_array_zero, discard_zero
from guppyalgos.utils import int_to_bits, qarray
from guppyalgos.algorithms.select.select_unary_iteration import (
    compute_cascade,
    adjacent_and,
    uncompute_cascade,
    _get_bools_and_diffs,
)


@guppy
@no_type_check
def accumulator_unary_iteration[
    n_s_q: nat,
    n_index_qubits: nat,
    n_index_elements: nat,
    CompAND: Callable[[qubit, qubit, qubit], None],
    UncompAND: Callable[[qubit, qubit, qubit], None],
    ControlledOp: Callable[[qubit, array[qubit, n_s_q]], None],
](
    controlled_ops: array[
        ControlledOp,
        n_index_elements,
    ],
    comp_and_op: CompAND,
    uncomp_and_op: UncompAND,
    index_qreg: array[qubit, n_index_qubits],
    state_qreg: array[qubit, n_s_q],
) -> None:
    """Apply controlled ops with accumulator.

    The index register is little-endian: ``index_qreg[0]`` is the
    least-significant qubit. The unary compute cascade runs from
    ``index_qreg[n_index_qubits - 1]`` toward ``index_qreg[0]`` and the
    uncompute cascade runs in the opposite direction.

    For an integer value `l` encoded in the index_qreg, the primitive applies
    user-defined controlled operations, controlled_ops[0] up to controlled_ops[l-1],
    to the state_qreg. Each controlled_ops[i] is applied as a 0-controlled operation
    from an accumulator qubit, whose state is flipped to ``|1>`` exactly once at index
    i = l via unary iteration.

    Disclaimer: This implementation is only valid for values l < n_index_elements
    on the index register and should NOT be used for out-of-range values
    l >= n_index_elements.

    The construction follows Fig. 8 of https://arxiv.org/pdf/1805.03662 and
    reuses the unary cascade logic from the select_unary_iteration routine
    implemented in guppyalgos. Work qubits and the accumulator qubit are
    discarded at the end of the procedure.

    Args:
        controlled_ops: The array of controlled operations.
        comp_and_op: The compute AND operation.
        uncomp_and_op: The uncompute AND operation.
        index_qreg: The little-endian index qubit register.
        state_qreg: The state qubit register.

    """
    index_bools, msb_diff_depths = _get_bools_and_diffs(
        n_index_qubits, n_index_elements
    )

    max_cascade_depth = len(index_qreg) - 1
    work_qreg = qarray(comptime(n_index_qubits - 1))
    n_data = n_index_elements

    acc_q = qubit()  # Accumulator qubit

    for i in range(n_data):
        if i == 0:
            compute_cascade(
                comp_and_op,
                max_cascade_depth,
                msb_diff_depths[i],
                index_bools[i],
                index_qreg,
                work_qreg,
            )

            # Flip accumulator to acc_q = |1> iff index == i
            cx(work_qreg[0], acc_q)

            # Apply controlled_ops[i] as a 0-controlled operation from acc_q
            x(acc_q)
            controlled_ops[i](acc_q, state_qreg)
            x(acc_q)

        else:
            if msb_diff_depths[i] != 0:
                adjacent_and(
                    msb_diff_depths[i],
                    index_qreg,
                    work_qreg,
                    index_bools[i],
                )

            if msb_diff_depths[i] != max_cascade_depth:
                compute_cascade(
                    comp_and_op,
                    max_cascade_depth,
                    msb_diff_depths[i],
                    index_bools[i],
                    index_qreg,
                    work_qreg,
                )

            # Flip accumulator to acc_q = |1> iff index == i
            cx(work_qreg[0], acc_q)

            # Apply controlled_ops[i] as a 0-controlled operation from acc_q
            x(acc_q)
            controlled_ops[i](acc_q, state_qreg)
            x(acc_q)

            # Uncompute work qubits at unary segment boundaries
            if msb_diff_depths[i] == max_cascade_depth or i == n_data - 1:
                next_msb_diff_depth = 0 if i == n_data - 1 else msb_diff_depths[i + 1]
                uncompute_cascade(
                    next_msb_diff_depth,
                    max_cascade_depth,
                    index_bools[i],
                    index_qreg,
                    work_qreg,
                    uncomp_and_op,
                )

    # Discard work register and accumulator qubit
    discard_array_zero(work_qreg)
    x(acc_q)  # Return acc_q to |0>
    discard_zero(acc_q)


# Comparator-guarded accumulator unary iteration implementation.
# The helpers below coherently XOR [index >= threshold] into the accumulator
# and uncompute the comparator workspace. The guarded accumulator starts from
# this flag, so out-of-range index values act trivially on the state register.
@guppy
@no_type_check
def _geq_const_compute[
    n_i_q: nat,
    n_w_q: nat,
    CompAND: Callable[[qubit, qubit, qubit], None],
    UncompAND: Callable[[qubit, qubit, qubit], None],
](
    comparison_depth: int,
    most_significant_index: int,
    threshold_bits: frozenarray[bool, n_i_q],
    index_qreg: array[qubit, n_i_q],
    target_q: qubit,
    work_qreg: array[qubit, n_w_q],
    comp_and_op: CompAND,
    uncomp_and_op: UncompAND,
) -> None:
    """Compute a comparator over little-endian index and threshold registers.

    ``threshold_bits[0]`` and ``index_qreg[0]`` are least significant. Comparison
    starts at position ``n_i_q - 1`` and proceeds toward position 0 so the first
    differing physical qubit is still the most-significant differing bit.

    Assumes:
      - threshold_bits[n_i_q - 1] is True
      - comparison_depth == 1 on the initial call

    After completion, work_qreg contains prefix-equality ancillas

        work[most_significant_index - j] =
            [x[n_i_q - 1:n_i_q - 2 - j] ==
             L[n_i_q - 1:n_i_q - 2 - j]]

    for j = 1, ..., most_significant_index.
    """
    if comparison_depth > most_significant_index:
        return

    physical_index = most_significant_index - comparison_depth
    work_index = most_significant_index - comparison_depth
    if comparison_depth == 1:
        # Strict-greater term at first possible differing bit:
        # [x[MSB] == L[MSB]] & [x[physical_index] == 1]
        # when L[physical_index] == 0.
        if not threshold_bits[physical_index]:
            term_q = temp_and_comp_index(
                index_qreg[most_significant_index],
                True,
                index_qreg[physical_index],
                True,
                comp_and_op,
            )
            cx(term_q, target_q)
            temp_and_uncomp_index(
                term_q,
                index_qreg[most_significant_index],
                True,
                index_qreg[physical_index],
                True,
                uncomp_and_op,
            )

        # Equality of the two most-significant bits.
        index_and(
            work_qreg[work_index],
            index_qreg[most_significant_index],
            True,
            index_qreg[physical_index],
            threshold_bits[physical_index],
            comp_and_op,
        )

    else:
        previous_work_index = work_index + 1

        # Strict-greater term at the most-significant differing bit:
        # [more-significant x bits == L bits] & [x[physical_index] == 1]
        # when L[physical_index] == 0.
        if not threshold_bits[physical_index]:
            term_q = temp_and_comp_index(
                work_qreg[previous_work_index],
                True,
                index_qreg[physical_index],
                True,
                comp_and_op,
            )
            cx(term_q, target_q)
            temp_and_uncomp_index(
                term_q,
                work_qreg[previous_work_index],
                True,
                index_qreg[physical_index],
                True,
                uncomp_and_op,
            )

        # Extend prefix equality toward the least-significant bit.
        index_and(
            work_qreg[work_index],
            work_qreg[previous_work_index],
            True,
            index_qreg[physical_index],
            threshold_bits[physical_index],
            comp_and_op,
        )

    _geq_const_compute(
        comparison_depth + 1,
        most_significant_index,
        threshold_bits,
        index_qreg,
        target_q,
        work_qreg,
        comp_and_op,
        uncomp_and_op,
    )


@guppy
@no_type_check
def _geq_const_uncompute[
    n_i_q: nat,
    n_w_q: nat,
](
    comparison_depth: int,
    threshold_bits: frozenarray[bool, n_i_q],
    index_qreg: array[qubit, n_i_q],
    work_qreg: array[qubit, n_w_q],
    uncomp_and_op: Callable[[qubit, qubit, qubit], None],
) -> None:
    """Uncompute the little-endian prefix-equality ladder toward the MSB."""
    most_significant_index = len(index_qreg) - 1
    physical_index = most_significant_index - comparison_depth
    work_index = most_significant_index - comparison_depth

    if comparison_depth == 1:
        index_and(
            work_qreg[work_index],
            index_qreg[most_significant_index],
            True,
            index_qreg[physical_index],
            threshold_bits[physical_index],
            uncomp_and_op,
        )
    else:
        previous_work_index = work_index + 1
        index_and(
            work_qreg[work_index],
            work_qreg[previous_work_index],
            True,
            index_qreg[physical_index],
            threshold_bits[physical_index],
            uncomp_and_op,
        )
    reset(work_qreg[work_index])

    if comparison_depth == 1:
        return

    _geq_const_uncompute(
        comparison_depth - 1,
        threshold_bits,
        index_qreg,
        work_qreg,
        uncomp_and_op,
    )


@guppy
@no_type_check
def _xor_geq_const_into[
    n_i_q: nat,
    n_w_q: nat,
    CompAND: Callable[[qubit, qubit, qubit], None],
    UncompAND: Callable[[qubit, qubit, qubit], None],
](
    target_q: qubit,
    threshold_bits: frozenarray[bool, n_i_q],
    comp_and_op: CompAND,
    uncomp_and_op: UncompAND,
    index_qreg: array[qubit, n_i_q],
    work_qreg: array[qubit, n_w_q],
) -> None:
    """XOR [index >= threshold] into target_q.

    ``threshold_bits`` is little-endian and its final (most-significant) bit is
    True.
    work_qreg is assumed to be empty and is returned empty after the comparator
    workspace is uncomputed.

    The equality case [index == threshold] is XORed into target_q after the
    strict-greater-than most-significant-difference terms.
    """
    most_significant_index = len(index_qreg) - 1

    _geq_const_compute(
        1,
        most_significant_index,
        threshold_bits,
        index_qreg,
        target_q,
        work_qreg,
        comp_and_op,
        uncomp_and_op,
    )

    # XOR equality case: index == threshold
    cx(work_qreg[0], target_q)

    _geq_const_uncompute(
        most_significant_index,
        threshold_bits,
        index_qreg,
        work_qreg,
        uncomp_and_op,
    )


@guppy
@no_type_check
def guarded_accumulator_unary_iteration[
    n_s_q: nat,
    n_index_qubits: nat,
    n_index_elements: nat,
    CompAND: Callable[[qubit, qubit, qubit], None],
    UncompAND: Callable[[qubit, qubit, qubit], None],
    ControlledOp: Callable[[qubit, array[qubit, n_s_q]], None],
](
    controlled_ops: array[
        ControlledOp,
        n_index_elements,
    ],
    comp_and_op: CompAND,
    uncomp_and_op: UncompAND,
    index_qreg: array[qubit, n_index_qubits],
    state_qreg: array[qubit, n_s_q],
) -> None:
    """Guarded accumulator unary iteration.

    The index register is little-endian: ``index_qreg[0]`` is the
    least-significant qubit. Both the comparator and unary compute cascade walk
    from ``index_qreg[n_index_qubits - 1]`` toward ``index_qreg[0]``; their
    uncompute passes walk in the opposite direction.

    For an integer value `l` encoded in the index_qreg, this behaves like
    accumulator_unary_iteration when 0 <= l < n_index_elements.

    For out-of-range values l >= n_index_elements on the index register, the
    state register is left unchanged.

    This is implemented by first XORing [index >= n_index_elements] into the
    accumulator qubit using a coherent comparator, then running the accumulator
    unary-iteration walk over the valid branch range only.

    If n_index_elements is a power of two, the guarded construction is identical
    to the unguarded one, so this function returns accumulator_unary_iteration.

    This variant is useful in tests and debugging when out-of-range values on
    the index register should act trivially on the state register.

    Args:
        controlled_ops: Array of functions acting on a control qubit and the
            state register.
        comp_and_op   : Logical AND operation used to build comparator terms
            and the unary tree.
        uncomp_and_op : Logical AND uncompute used for comparator terms and
            cascade uncomputation.
        index_qreg    : Indexing quantum register.
        state_qreg    : Quantum register on which the controlled_ops act.

    """
    if n_index_elements == 2**n_index_qubits:
        accumulator_unary_iteration(
            controlled_ops, comp_and_op, uncomp_and_op, index_qreg, state_qreg
        )
        return
    threshold_bits = comptime(
        int_to_bits(n_index_elements % 2**n_index_qubits, n_index_qubits)
    )

    index_bools, msb_diff_depths = _get_bools_and_diffs(
        n_index_qubits, n_index_elements
    )

    max_cascade_depth = len(index_qreg) - 1
    work_qreg = qarray(comptime(n_index_qubits - 1))
    n_data = n_index_elements

    acc_q = qubit()

    # Initialize acc_q = |1> iff index >= n_index_elements
    _xor_geq_const_into(
        acc_q,
        threshold_bits,
        comp_and_op,
        uncomp_and_op,
        index_qreg,
        work_qreg,
    )

    for i in range(n_data):
        if i == 0:
            compute_cascade(
                comp_and_op,
                max_cascade_depth,
                msb_diff_depths[i],
                index_bools[i],
                index_qreg,
                work_qreg,
            )

            # XOR [index == i] into acc_q
            cx(work_qreg[0], acc_q)

            # Apply controlled_ops[i] as a 0-controlled operation from acc_q
            x(acc_q)
            controlled_ops[i](acc_q, state_qreg)
            x(acc_q)

        else:
            if msb_diff_depths[i] != 0:
                adjacent_and(
                    msb_diff_depths[i],
                    index_qreg,
                    work_qreg,
                    index_bools[i],
                )

            if msb_diff_depths[i] != max_cascade_depth:
                compute_cascade(
                    comp_and_op,
                    max_cascade_depth,
                    msb_diff_depths[i],
                    index_bools[i],
                    index_qreg,
                    work_qreg,
                )

            # XOR [index == i] into acc_q
            cx(work_qreg[0], acc_q)

            # Apply controlled_ops[i] as a 0-controlled operation from acc_q
            x(acc_q)
            controlled_ops[i](acc_q, state_qreg)
            x(acc_q)

            if msb_diff_depths[i] == max_cascade_depth or i == n_data - 1:
                next_msb_diff_depth = 0 if i == n_data - 1 else msb_diff_depths[i + 1]
                uncompute_cascade(
                    next_msb_diff_depth,
                    max_cascade_depth,
                    index_bools[i],
                    index_qreg,
                    work_qreg,
                    uncomp_and_op,
                )

    # Discard work register and accumulator qubit
    discard_array_zero(work_qreg)
    x(acc_q)  # Return acc_q to |0>
    discard_zero(acc_q)
