"""Meta function to create a unary iteration Select function."""

from collections.abc import Callable
from math import ceil, log2
from typing import no_type_check

from guppylang import comptime, guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import Function, array, nat
from guppylang.std.quantum import cx, qubit, reset, x, discard_array

from guppyalgos.primitives.gate_decompositions.and_op import (
    index_and,
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.utils import int_to_bits, qarray


def get_index_bools(
    n_index_qubits: int,
    n_index_elements: int,
    index_fn: Callable[[int, int], list[bool]],
) -> list[list[bool]]:
    """Return the index boolean list.

    Args:
        n_index_qubits: The number of index qubits.
        n_index_elements: The number of index elements.
        index_fn: The indexing function. Unary iteration passes
            :func:`int_to_bits`, so position 0 is the
            least-significant bit.

    Returns:
        list[list[bool]]: A list of boolean values for indexing.

    """
    return [index_fn(i, n_index_qubits) for i in range(n_index_elements)]


def msb_diff_depths(bools: list[list[bool]]) -> list[int]:
    """Return cascade depths of the most-significant differing bits.

    The input bitstrings are little-endian, so the most-significant bit is at the
    end of each array. The scan therefore begins at position ``len(bits) - 1``
    and proceeds toward position 0. Returned values are cascade depths: depth 0
    corresponds to physical position ``len(bits) - 1``, depth 1 to position
    ``len(bits) - 2``, and so on. The first value is 0 because the initial compute
    cascade begins at the most-significant qubit.

    Args:
        bools (list[list[bool]]): A list of lists containing boolean values.

    Returns:
        Cascade depths of the most-significant differing boolean values.

    """
    diff_depths = [0]
    for i in range(1, len(bools)):
        previous = bools[i - 1]
        current = bools[i]
        for cascade_depth, (previous_bit, current_bit) in enumerate(
            zip(reversed(previous), reversed(current), strict=True)
        ):
            if previous_bit != current_bit:
                diff_depths.append(cascade_depth)
                break

    return diff_depths


@guppy
@no_type_check
def compute_cascade[n_i_q: nat, n_w_q: nat](
    comp_and_op: Callable[[qubit, qubit, qubit], None],
    max_cascade_depth: int,
    msb_diff_depth: int,
    bools: array[bool, n_i_q],
    index_qreg: array[qubit, n_i_q],
    work_qreg: array[qubit, n_w_q],
) -> None:
    """Compute a little-endian unary cascade from a high index toward qubit 0.

    A full compute starts at ``index_qreg[n_i_q - 1]`` and proceeds toward
    ``index_qreg[0]``. After an adjacent-AND transition, computation resumes at
    the intermediate level given by ``msb_diff_depth`` and still proceeds toward
    qubit 0.

    Args:
        comp_and_op: The AND operation used to compute each work qubit.
        max_cascade_depth: The deepest cascade depth, equal to the final physical
            index of ``index_qreg``.
        msb_diff_depth: The cascade depth of the most-significant differing bit
            at which computation starts. Depth 0 corresponds to
            ``index_qreg[n_i_q - 1]``.
        bools: The little-endian boolean values for the current index.
        index_qreg: The little-endian index qubit register.
        work_qreg: The work-qubit array. Position 0 is the final cascade target
            associated with ``index_qreg[0]``.

    """
    if msb_diff_depth >= max_cascade_depth:
        return

    next_cascade_depth = msb_diff_depth + 1
    work_index = max_cascade_depth - next_cascade_depth
    if msb_diff_depth == 0:
        index_and(
            work_qreg[work_index],
            index_qreg[max_cascade_depth],
            bools[max_cascade_depth],
            index_qreg[max_cascade_depth - 1],
            bools[max_cascade_depth - 1],
            comp_and_op,
        )
    else:
        previous_work_index = work_index + 1
        physical_index = max_cascade_depth - next_cascade_depth
        index_and(
            work_qreg[work_index],
            work_qreg[previous_work_index],
            True,
            index_qreg[physical_index],
            bools[physical_index],
            comp_and_op,
        )

    compute_cascade(
        comp_and_op,
        max_cascade_depth,
        next_cascade_depth,
        bools,
        index_qreg,
        work_qreg,
    )


@guppy
@no_type_check
def adjacent_and[n_w_q: nat, n_i_q: nat](
    msb_diff_depth: int,
    index_qreg: array[qubit, n_i_q],
    work_qreg: array[qubit, n_w_q],
    bools: array[bool, n_i_q],
) -> None:
    """Update a little-endian unary cascade for the next adjacent index.

    The work array follows the little-endian index positions: computation moves
    from its highest position toward ``work_qreg[0]``. Consequently, the deepest
    adjacent AND targets position 0, which incorporates the least-significant
    index qubit.

    Args:
        msb_diff_depth: The cascade depth of the most-significant difference
            between the previous and current index values.
        index_qreg: The little-endian index qubit register.
        work_qreg: The work-qubit array, ordered in the same direction as the
            little-endian index register.
        bools: The little-endian boolean values for the current index.

    """
    max_cascade_depth = len(index_qreg) - 1
    target_work_index = max_cascade_depth - msb_diff_depth
    if msb_diff_depth == 1:
        most_significant_index = len(index_qreg) - 1
        if not bools[most_significant_index]:
            x(index_qreg[most_significant_index])
        cx(index_qreg[most_significant_index], work_qreg[target_work_index])
        if not bools[most_significant_index]:
            x(index_qreg[most_significant_index])
    else:
        previous_work_index = target_work_index + 1
        cx(work_qreg[previous_work_index], work_qreg[target_work_index])


@guppy
@no_type_check
def uncompute_cascade[n_i_q: nat, n_w_q: nat](
    next_msb_diff_depth: int,
    current_cascade_depth: int,
    bools: array[bool, n_i_q],
    index_qreg: array[qubit, n_i_q],
    work_qreg: array[qubit, n_w_q],
    uncomp_and_op: Callable[[qubit, qubit, qubit], None],
) -> None:
    """Uncompute a little-endian unary cascade from qubit 0 toward a high index.

    This is the opposite traversal to :func:`compute_cascade`: work qubits are
    removed starting at the ``index_qreg[0]`` level and proceeding toward
    ``index_qreg[n_i_q - 1]`` or the intermediate level
    ``next_msb_diff_depth``.

    Args:
        next_msb_diff_depth: The cascade depth at which uncomputation stops.
            Depth 0 uncomputes the full cascade.
        current_cascade_depth: The current deepest computed cascade depth.
        bools: The little-endian boolean values used to compute the work qubits.
        index_qreg: The little-endian index qubit register.
        work_qreg: The work-qubit array, uncomputed from position 0 upward.
        uncomp_and_op: The AND operation used to uncompute each work qubit.

    """
    if current_cascade_depth <= next_msb_diff_depth:
        return

    most_significant_index = len(index_qreg) - 1
    work_index = most_significant_index - current_cascade_depth
    if current_cascade_depth == 1:
        index_and(
            work_qreg[work_index],
            index_qreg[most_significant_index],
            bools[most_significant_index],
            index_qreg[most_significant_index - 1],
            bools[most_significant_index - 1],
            uncomp_and_op,
        )
    else:
        previous_work_index = work_index + 1
        physical_index = most_significant_index - current_cascade_depth
        index_and(
            work_qreg[work_index],
            work_qreg[previous_work_index],
            True,
            index_qreg[physical_index],
            bools[physical_index],
            uncomp_and_op,
        )
    reset(work_qreg[work_index])

    uncompute_cascade(
        next_msb_diff_depth,
        current_cascade_depth - 1,
        bools,
        index_qreg,
        work_qreg,
        uncomp_and_op,
    )


@guppy.comptime
@no_type_check
def _get_bools_and_diffs(
    n_index_q: nat @ comptime,
    n_index_el: nat @ comptime,
) -> tuple[
    array[array[bool, "n_index_q"], "n_index_el"],
    array[int, "n_index_el"],
]:
    """Compute little-endian bitstrings and right-to-left cascade differences.

    Args:
        n_index_q: The number of index qubits.
        n_index_el: The number of index values visited by unary iteration.

    Returns:
        The little-endian index bitstrings and their most-significant-difference
        cascade depths.

    """
    if n_index_q < ceil(log2(n_index_el)):
        raise ValueError(
            f"{n_index_q} qubits is not enough to iterate over {n_index_el} terms"
        )
    bools = get_index_bools(n_index_q, n_index_el, int_to_bits)
    diffs = msb_diff_depths(bools)
    return bools, diffs


@guppy
@no_type_check
def select_unary_iteration[
    TargetRegs,
    n_index_el: nat,
    n_index_qubits: nat,
    CompAND: Callable[[qubit, qubit, qubit], None],
    UncompAND: Callable[[qubit, qubit, qubit], None],
](
    controlled_ops: array[
        Callable[[qubit, TargetRegs], None],
        n_index_el,
    ],
    comp_and_op: CompAND,
    uncomp_and_op: UncompAND,
    index_qreg: array[qubit, n_index_qubits],
    state_qreg: TargetRegs,
) -> None:
    """Unary iteration select function.

    This function performs the unary iteration select operation.
    The index register is little-endian: ``index_qreg[0]`` is the
    least-significant qubit, and integer index ``i`` selects ``controlled_ops[i]``.
    There is also freedom to choose the compute and uncompute AND operations.
    However it is up to the user to ensure that the AND operations
    are correct.

    Algorithmic workflow:

    1. Compute the index boolean values and most-significant differing bits.
    2. Iterate over each index element:

       a. Compute a little-endian AND cascade generating the work qubits at each
          step in the recursion.
       b. Apply the corresponding controlled operation to the state register.
       c. For subsequent index elements, perform adjacent AND operations if the
          most-significant differing bit is not the last index.
       d. Finally, uncompute the little-endian AND cascade to the most-significant
          differing bit between ``i`` and ``i + 1``.

    3. Discard the work qubit register.

    Args:
        controlled_ops: The array of controlled operations.
        comp_and_op: The compute AND operation.
        uncomp_and_op: The uncompute AND operation.
        index_qreg: The index qubit register.
        state_qreg: The state qubit register.

    """
    index_bools, msb_diff_depths = _get_bools_and_diffs(n_index_qubits, n_index_el)

    max_cascade_depth = len(index_qreg) - 1
    work_qreg = qarray(comptime(n_index_qubits - 1))
    n_iterations = n_index_el

    for iteration_index in range(n_iterations):
        if iteration_index == 0:
            compute_cascade(
                comp_and_op,
                max_cascade_depth,
                msb_diff_depths[iteration_index],
                index_bools[iteration_index],
                index_qreg,
                work_qreg,
            )
        else:
            if msb_diff_depths[iteration_index] != 0:
                adjacent_and(
                    msb_diff_depths[iteration_index],
                    index_qreg,
                    work_qreg,
                    index_bools[iteration_index],
                )

            if msb_diff_depths[iteration_index] != max_cascade_depth:
                compute_cascade(
                    comp_and_op,
                    max_cascade_depth,
                    msb_diff_depths[iteration_index],
                    index_bools[iteration_index],
                    index_qreg,
                    work_qreg,
                )

        # The final work qubit is the AND target at the index_qreg[0] level.
        controlled_ops[iteration_index](work_qreg[0], state_qreg)

        next_msb_diff_depth = (
            0
            if iteration_index == n_iterations - 1
            else msb_diff_depths[iteration_index + 1]
        )
        uncompute_cascade(
            next_msb_diff_depth,
            max_cascade_depth,
            index_bools[iteration_index],
            index_qreg,
            work_qreg,
            uncomp_and_op,
        )

    discard_array(work_qreg)


@guppy.comptime
@no_type_check
def _get_cntrl_bools_and_diffs(
    n_index_q: nat @ comptime,
    n_index_el: nat @ comptime,
) -> tuple[
    array[array[bool, "n_index_q"], "n_index_el"],
    array[int, "n_index_el"],
]:
    """Compute bitstrings and cascade differences for controlled iteration.

    Args:
        n_index_q: The number of index qubits.
        n_index_el: The number of index values visited by unary iteration.

    Returns:
        The little-endian index bitstrings and their most-significant-difference
        cascade depths, including the external-control level.

    """
    if n_index_q < ceil(log2(n_index_el)):
        raise ValueError(
            f"{n_index_q} qubits is not enough to iterate over {n_index_el}\
            terms"
        )
    bools = get_index_bools(n_index_q, n_index_el, int_to_bits)
    register_diff_depths = msb_diff_depths(bools)
    # The external control is the first level of the controlled cascade, so each
    # subsequent register difference is one level deeper.
    diffs = [0] + [diff + 1 for diff in register_diff_depths[1:]]
    return bools, diffs


@guppy
@no_type_check
def cntrl_select_unary_iteration[
    TargetRegs,
    n_index_qubits: nat,
    n_index_elements: nat,
    CompAND: Callable[[qubit, qubit, qubit], None],
    UncompAND: Callable[[qubit, qubit, qubit], None],
](
    controlled_ops: array[
        Callable[[qubit, TargetRegs], None],
        n_index_elements,
    ],
    comp_and_op: CompAND,
    uncomp_and_op: UncompAND,
    control: qubit,
    index_qreg: array[qubit, n_index_qubits],
    state_qreg: TargetRegs,
) -> None:
    """Unary iteration select function.

    This function performs the unary iteration select operation.
    The index register is little-endian: ``index_qreg[0]`` is the
    least-significant qubit, and integer index ``i`` selects ``controlled_ops[i]``.
    There is also freedom to choose the compute and uncompute
    AND operations. However it is up to the user to ensure that the AND operations
    are correct.

    Algorithmic workflow:

    1. Compute the index boolean values and most-significant differing bits.
    2. Iterate over each index element:

       a. Compute a little-endian AND cascade generating the work qubits at each
          step in the recursion.
       b. Apply the corresponding controlled operation to the state register.
       c. For subsequent index elements, perform adjacent AND operations if the
          most-significant differing bit is not the last index.
       d. Finally, uncompute the little-endian AND cascade to the most-significant
          differing bit between ``i`` and ``i + 1``.

    3. Discard the work qubit register.

    Args:
        controlled_ops: The array of controlled operations.
        comp_and_op: The compute AND operation.
        uncomp_and_op: The uncompute AND operation.
        control: The control qubit for the controlled version of the select unary
            iteration.
        index_qreg: The index qubit register.
        state_qreg: The state qubit register.

    """
    index_bools, msb_diff_depths = _get_cntrl_bools_and_diffs(
        n_index_qubits,
        n_index_elements,
    )

    max_cascade_depth = len(index_qreg)
    work_qreg = qarray(comptime(n_index_qubits))
    n_iterations = n_index_elements

    for iteration_index in range(n_iterations):
        if iteration_index == 0:
            compute_cntrl_cascade(
                comp_and_op,
                max_cascade_depth,
                msb_diff_depths[iteration_index],
                index_bools[iteration_index],
                control,
                index_qreg,
                work_qreg,
            )
        else:
            if msb_diff_depths[iteration_index] != 0:
                cntrl_adjacent_and(msb_diff_depths[iteration_index], control, work_qreg)

            if msb_diff_depths[iteration_index] != max_cascade_depth:
                compute_cntrl_cascade(
                    comp_and_op,
                    max_cascade_depth,
                    msb_diff_depths[iteration_index],
                    index_bools[iteration_index],
                    control,
                    index_qreg,
                    work_qreg,
                )

        # The final work qubit is the AND target at the index_qreg[0] level.
        controlled_ops[iteration_index](work_qreg[0], state_qreg)

        next_msb_diff_depth = (
            0
            if iteration_index == n_iterations - 1
            else msb_diff_depths[iteration_index + 1]
        )
        uncompute_cntrl_cascade(
            next_msb_diff_depth,
            max_cascade_depth,
            index_bools[iteration_index],
            control,
            index_qreg,
            work_qreg,
            uncomp_and_op,
        )

    discard_array(work_qreg)


@guppy
@no_type_check
def compute_cntrl_cascade[n_i_q: nat, n_w_q: nat](
    comp_and_op: Callable[[qubit, qubit, qubit], None],
    max_cascade_depth: int,
    msb_diff_depth: int,
    bools: array[bool, n_i_q],
    control_q: qubit,
    index_qreg: array[qubit, n_i_q],
    work_qreg: array[qubit, n_w_q],
) -> None:
    """Compute a controlled little-endian cascade toward index qubit 0.

    The cascade starts with the external control and the most-significant index
    qubit, then proceeds toward ``index_qreg[0]``.

    The work positions are held in an array. Computation fills positions from
    the highest required index toward ``work_qreg[0]``.

    Args:
        comp_and_op: The AND operation used to compute each work qubit.
        max_cascade_depth: The deepest controlled-cascade depth.
        msb_diff_depth: The cascade depth of the most-significant differing bit
            at which computation starts.
        bools: The little-endian boolean values for the current index.
        control_q: The external control qubit.
        index_qreg: The little-endian index qubit register.
        work_qreg: The work-qubit array. Position 0 is the final cascade target
            associated with ``index_qreg[0]``.

    """
    if msb_diff_depth >= max_cascade_depth:
        return

    next_cascade_depth = msb_diff_depth + 1
    work_index = max_cascade_depth - next_cascade_depth
    if msb_diff_depth == 0:
        most_significant_index = len(index_qreg) - 1
        index_and(
            work_qreg[work_index],
            control_q,
            True,
            index_qreg[most_significant_index],
            bools[most_significant_index],
            comp_and_op,
        )

    else:
        previous_work_index = work_index + 1
        physical_index = len(index_qreg) - next_cascade_depth
        index_and(
            work_qreg[work_index],
            work_qreg[previous_work_index],
            True,
            index_qreg[physical_index],
            bools[physical_index],
            comp_and_op,
        )

    compute_cntrl_cascade(
        comp_and_op,
        max_cascade_depth,
        next_cascade_depth,
        bools,
        control_q,
        index_qreg,
        work_qreg,
    )


@guppy
@no_type_check
def cntrl_adjacent_and[n_w_q: nat](
    msb_diff_depth: int,
    control_q: qubit,
    work_qreg: array[qubit, n_w_q],
) -> None:
    """Perform indexed adjacent AND operations via a CNOT.

    Using the identity from Fig 6. https://arxiv.org/pdf/1805.03662
    where two adjacent AND operations can be performed with a CNOT
    between the first control and the target.

    If the most-significant differing bit has depth 1, the first AND operation
    is performed between the first index qubit and the first work qubit.
    Otherwise the AND operation is performed between the previous work qubit
    and the current work qubit.

    Args:
        msb_diff_depth: The controlled-cascade depth of the most-significant
            difference.
        control_q: The external control qubit.
        work_qreg: The work-qubit array, ordered in the same direction as the
            little-endian index register.

    """
    target_work_index = len(work_qreg) - msb_diff_depth
    if msb_diff_depth == 1:
        cx(control_q, work_qreg[target_work_index])
    else:
        previous_work_index = target_work_index + 1
        cx(work_qreg[previous_work_index], work_qreg[target_work_index])


@guppy
@no_type_check
def uncompute_cntrl_cascade[n_i_q: nat, n_w_q: nat](
    next_msb_diff_depth: int,
    current_cascade_depth: int,
    bools: array[bool, n_i_q],
    control_q: qubit,
    index_qreg: array[qubit, n_i_q],
    work_qreg: array[qubit, n_w_q],
    uncomp_and_op: Callable[[qubit, qubit, qubit], None],
) -> None:
    """Uncompute a controlled little-endian cascade away from index qubit 0.

    The cascade unwinds from ``index_qreg[0]`` toward the most-significant index
    qubit and the external control.

    ``next_msb_diff_depth`` determines where little-endian uncomputation stops.

    The work qubits are uncomputed from position 0 toward higher array positions.

    Args:
        next_msb_diff_depth: The controlled-cascade depth at which uncomputation
            stops.
        current_cascade_depth: The current deepest computed cascade depth.
        bools: The little-endian boolean values used to compute the work qubits.
        control_q: The external control qubit.
        index_qreg: The little-endian index qubit register.
        work_qreg: The work-qubit array, uncomputed from position 0 upward.
        uncomp_and_op: The AND operation used to uncompute each work qubit.

    """
    if current_cascade_depth <= next_msb_diff_depth:
        return

    work_index = len(index_qreg) - current_cascade_depth
    if current_cascade_depth == 1:
        most_significant_index = len(index_qreg) - 1
        index_and(
            work_qreg[work_index],
            control_q,
            True,
            index_qreg[most_significant_index],
            bools[most_significant_index],
            uncomp_and_op,
        )
    else:
        previous_work_index = work_index + 1
        physical_index = len(index_qreg) - current_cascade_depth
        index_and(
            work_qreg[work_index],
            work_qreg[previous_work_index],
            True,
            index_qreg[physical_index],
            bools[physical_index],
            uncomp_and_op,
        )
    reset(work_qreg[work_index])

    uncompute_cntrl_cascade(
        next_msb_diff_depth,
        current_cascade_depth - 1,
        bools,
        control_q,
        index_qreg,
        work_qreg,
        uncomp_and_op,
    )


def build_select_unary_from_data[Data, TargetRegs, n_i_q: nat](
    data_input: list[Data],
    data_to_ctrl_op: Callable[
        [Data], GuppyFunctionDefinition[[qubit, TargetRegs], None]
    ],
    comp_and_op: GuppyFunctionDefinition[
        [qubit, qubit, qubit], None
    ] = temp_and_compute,
    uncomp_and_op: GuppyFunctionDefinition[
        [qubit, qubit, qubit], None
    ] = temp_and_uncompute,
) -> GuppyFunctionDefinition[[array[qubit, n_i_q], TargetRegs], None]:
    """Construct a unary iteration over unitaries constructed from Python data.

    Args:
        data_input (list[Data]): Classical data to construct controlled unitaries.
        data_to_ctrl_op: Method to create controlled unitaries from data.
        comp_and_op: Function to compute temporary AND operations.
        uncomp_and_op: Function to uncompute temporary AND operations.

    Returns:
        GuppyFunctionDefinition: The unary iteration applying the select using unitaries
        constructed from the given data.

    """
    n_data_elements = len(data_input)
    n_index_qubits = ceil(log2(n_data_elements))

    @guppy.comptime
    @no_type_check
    def build_cntrl_ops_from_data[TargetRegs]() -> array[
        Function[[qubit, TargetRegs], None],
        n_data_elements,
    ]:
        """Build controlled operations from classical data.

        This loops over the classical data and builds a list of ops
        using the provided method for each data entry.
        """
        return [data_to_ctrl_op(data) for data in data_input]

    @guppy
    @no_type_check
    def unary_iteration_fn[TargetRegs](
        index_qreg: array[qubit, n_index_qubits],
        target_regs: TargetRegs,
    ) -> None:
        """Resulting unary iteration function.

        This is the constructed unary iteration function that takes
        in the index and state qubit registers and applies the select.

        Args:
            index_qreg (array[qubit, n_index_qubits]): The
                index qubit register.
            target_regs (TargetRegs): Registers for the applied ops.

        """
        controlled_ops = build_cntrl_ops_from_data[TargetRegs]()
        select_unary_iteration(
            controlled_ops,
            comp_and_op,
            uncomp_and_op,
            index_qreg,
            target_regs,
        )

    return unary_iteration_fn


def build_cntrl_select_unary_from_data[Data, TargetRegs, n_i_q: nat](
    data_input: list[Data],
    data_to_ctrl_op: Callable[
        [Data], GuppyFunctionDefinition[[qubit, TargetRegs], None]
    ],
    comp_and_op: GuppyFunctionDefinition[
        [qubit, qubit, qubit], None
    ] = temp_and_compute,
    uncomp_and_op: GuppyFunctionDefinition[
        [qubit, qubit, qubit], None
    ] = temp_and_uncompute,
) -> GuppyFunctionDefinition[[qubit, array[qubit, n_i_q], TargetRegs], None]:
    """Construct a unary iteration over unitaries constructed from Python data.

    Args:
        data_input (list[Data]): Classical data to construct controlled unitaries.
        data_to_ctrl_op: Method to create controlled unitaries from data.
        comp_and_op: Function to compute temporary AND operations.
        uncomp_and_op: Function to uncompute temporary AND operations.

    Returns:
        GuppyFunctionDefinition: The unary iteration applying the select using unitaries
        constructed from the given data.

    """
    n_data_elements = len(data_input)
    n_index_qubits = ceil(log2(n_data_elements))

    @guppy.comptime
    @no_type_check
    def build_cntrl_ops_from_data[TargetRegs]() -> array[
        Function[[qubit, TargetRegs], None],
        n_data_elements,
    ]:
        """Build controlled operations from classical data.

        This loops over the classical data and builds a list of operations
        using the provided method for each data entry.
        """
        return [data_to_ctrl_op(data) for data in data_input]

    @guppy
    @no_type_check
    def unary_iteration_fn[TargetRegs](
        control: qubit,
        index_qreg: array[qubit, n_index_qubits],
        target_regs: TargetRegs,
    ) -> None:
        """Resulting unary iteration select.

        This is the constructed unary iteration function that takes
        in the index and state qubit registers and applies the select operation.

        Args:
            control (qubit): control qubit
            index_qreg (array[qubit, n_index_qubits]): The
                index qubit register.
            target_regs (TargetRegs): The
                state qubit register.

        """
        controlled_ops = build_cntrl_ops_from_data[TargetRegs]()
        cntrl_select_unary_iteration(
            controlled_ops,
            comp_and_op,
            uncomp_and_op,
            control,
            index_qreg,
            target_regs,
        )

    return unary_iteration_fn
