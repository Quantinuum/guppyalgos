"""Meta function to create a unary iteration Select function."""

from collections.abc import Callable
from math import ceil, log2
from typing import no_type_check

from guppylang import comptime, guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import Function, Unitary, array, control, dagger, nat
from guppylang.std.quantum import cx, qubit, reset, x, discard_array

from guppyalgos.primitives.gate_decompositions.and_op import (
    and_pattern_controlled,
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.utils import qarray
from guppyalgos.utils import int_to_bits


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


@guppy(controllable=True)
@no_type_check
def _apply_cascade_and_dagger[AndOp: Unitary[[qubit, qubit, qubit], None]](
    and_op: AndOp,
    target: qubit,
    prefix: qubit,
    prefix_bit: bool,
    index: qubit,
    index_bit: bool,
) -> None:
    """Apply the patterned AND's dagger."""
    with dagger:
        and_pattern_controlled(target, prefix, prefix_bit, index, index_bit, and_op)


@guppy.comptime(controllable=True)
@no_type_check
def _apply_cascade_and[AndOp: Unitary[[qubit, qubit, qubit], None]](
    and_op: AndOp,
    target: qubit,
    prefix: qubit,
    prefix_bit: bool,
    index: qubit,
    index_bit: bool,
    inverse: bool @ comptime,
) -> None:
    """Trace only the requested mode, since cleanup callbacks may measure."""
    if inverse:
        _apply_cascade_and_dagger(and_op, target, prefix, prefix_bit, index, index_bit)
    else:
        and_pattern_controlled(target, prefix, prefix_bit, index, index_bit, and_op)


@guppy(controllable=True)
@no_type_check
def _cascade_step[n_i_q: nat, n_w_q: nat, AndOp: Unitary[[qubit, qubit, qubit], None]](
    and_op: AndOp,
    most_significant_index: int,
    work_index: int,
    bools: array[bool, n_i_q],
    index_qreg: array[qubit, n_i_q],
    work_qreg: array[qubit, n_w_q],
    inverse: bool @ comptime,
) -> None:
    """Use the top index bit or the previous work flag as the prefix."""
    if work_index == most_significant_index - 1:
        _apply_cascade_and(
            and_op,
            work_qreg[work_index],
            index_qreg[most_significant_index],
            bools[most_significant_index],
            index_qreg[work_index],
            bools[work_index],
            inverse,
        )
    else:
        _apply_cascade_and(
            and_op,
            work_qreg[work_index],
            work_qreg[work_index + 1],
            True,
            index_qreg[work_index],
            bools[work_index],
            inverse,
        )


@guppy(controllable=True)
@no_type_check
def _dagger_cascade[
    n_i_q: nat,
    n_w_q: nat,
    AndOp: Unitary[[qubit, qubit, qubit], None],
](
    comp_and_op: AndOp,
    max_cascade_depth: int,
    msb_diff_depth: int,
    bools: array[bool, n_i_q],
    index_qreg: array[qubit, n_i_q],
    work_qreg: array[qubit, n_w_q],
) -> None:
    """Apply the AND daggers from the lowest work index up."""
    for work_index in range(max_cascade_depth - msb_diff_depth):
        _cascade_step(
            comp_and_op,
            max_cascade_depth,
            work_index,
            bools,
            index_qreg,
            work_qreg,
            True,
        )


@guppy.unitary
class compute_cascade:
    """Compute a little-endian unary cascade from a high index toward qubit 0.

    A full compute starts at ``index_qreg[n_i_q - 1]``. After an adjacent-AND
    transition, computation resumes below the existing prefix at
    ``msb_diff_depth``. Position 0 of the work register is the final target.

    Args:
        comp_and_op: The AND operation used to compute each work qubit.
        max_cascade_depth: The deepest cascade depth, equal to the final physical
            index of ``index_qreg``.
        msb_diff_depth: The depth at which computation starts. Depth 0
            corresponds to ``index_qreg[n_i_q - 1]``.
        bools: The little-endian boolean values for the current index.
        index_qreg: The little-endian index qubit register.
        work_qreg: The work qubits, ordered from the lowest index position up.

    """

    @guppy(controllable=True)
    @no_type_check
    def __call__[n_i_q: nat, n_w_q: nat, AndOp: Unitary[[qubit, qubit, qubit], None]](
        comp_and_op: AndOp,
        max_cascade_depth: int,
        msb_diff_depth: int,
        bools: array[bool, n_i_q],
        index_qreg: array[qubit, n_i_q],
        work_qreg: array[qubit, n_w_q],
    ) -> None:
        """Apply the AND steps from the highest work index down."""
        for work_index in range(max_cascade_depth - msb_diff_depth - 1, -1, -1):
            _cascade_step(
                comp_and_op,
                max_cascade_depth,
                work_index,
                bools,
                index_qreg,
                work_qreg,
                False,
            )

    @guppy
    @no_type_check
    def daggered[n_i_q: nat, n_w_q: nat, AndOp: Unitary[[qubit, qubit, qubit], None]](
        comp_and_op: AndOp,
        max_cascade_depth: int,
        msb_diff_depth: int,
        bools: array[bool, n_i_q],
        index_qreg: array[qubit, n_i_q],
        work_qreg: array[qubit, n_w_q],
    ) -> None:
        """Apply the dagger of each AND step in reverse order."""
        _dagger_cascade(
            comp_and_op, max_cascade_depth, msb_diff_depth, bools, index_qreg, work_qreg
        )

    @guppy
    @no_type_check
    def ctrl_daggered[
        n_i_q: nat,
        n_w_q: nat,
        AndOp: Unitary[[qubit, qubit, qubit], None],
        n_controls: nat,
    ](
        comp_and_op: AndOp,
        max_cascade_depth: int,
        msb_diff_depth: int,
        bools: array[bool, n_i_q],
        index_qreg: array[qubit, n_i_q],
        work_qreg: array[qubit, n_w_q],
        controls: array[qubit, n_controls],
    ) -> None:
        """Apply the AND daggers in reverse order under the additional controls."""
        with control(controls):
            _dagger_cascade(
                comp_and_op,
                max_cascade_depth,
                msb_diff_depth,
                bools,
                index_qreg,
                work_qreg,
            )


@guppy.unitary
class adjacent_and:
    """Update a little-endian unary cascade for the next adjacent index.

    Flip the work qubit at the differing depth conditioned on its unchanged
    prefix. The update is self-adjoint for the same prefix. Position 0 of the
    work array corresponds to the least-significant index qubit.

    Args:
        msb_diff_depth: The cascade depth of the most-significant difference
            between the previous and current index values.
        index_qreg: The little-endian index qubit register.
        work_qreg: The work-qubit array, ordered like the index register.
        bools: The little-endian boolean values for the current index.

    """

    @guppy(controllable=True)
    @no_type_check
    def __call__[n_w_q: nat, n_i_q: nat](
        msb_diff_depth: int,
        index_qreg: array[qubit, n_i_q],
        work_qreg: array[qubit, n_w_q],
        bools: array[bool, n_i_q],
    ) -> None:
        """Flip the work flag using a signed index bit or a work prefix."""
        most_significant_index = comptime(n_i_q - 1)
        target_work_index = most_significant_index - msb_diff_depth
        if msb_diff_depth == 1:
            if not bools[most_significant_index]:
                x(index_qreg[most_significant_index])
            cx(index_qreg[most_significant_index], work_qreg[target_work_index])
            if not bools[most_significant_index]:
                x(index_qreg[most_significant_index])
        else:
            cx(work_qreg[target_work_index + 1], work_qreg[target_work_index])

    @guppy
    @no_type_check
    def daggered[n_w_q: nat, n_i_q: nat](
        msb_diff_depth: int,
        index_qreg: array[qubit, n_i_q],
        work_qreg: array[qubit, n_w_q],
        bools: array[bool, n_i_q],
    ) -> None:
        """Reuse the self-adjoint adjacent update."""
        adjacent_and(msb_diff_depth, index_qreg, work_qreg, bools)

    @guppy
    @no_type_check
    def ctrl_daggered[n_w_q: nat, n_i_q: nat, n_controls: nat](
        msb_diff_depth: int,
        index_qreg: array[qubit, n_i_q],
        work_qreg: array[qubit, n_w_q],
        bools: array[bool, n_i_q],
        controls: array[qubit, n_controls],
    ) -> None:
        """Control the self-adjoint adjacent update."""
        with control(controls):
            adjacent_and(msb_diff_depth, index_qreg, work_qreg, bools)


@guppy
@no_type_check
def uncompute_cascade[
    n_i_q: nat,
    n_w_q: nat,
    UncompAND: Unitary[[qubit, qubit, qubit], None],
](
    next_msb_diff_depth: int,
    current_cascade_depth: int,
    bools: array[bool, n_i_q],
    index_qreg: array[qubit, n_i_q],
    work_qreg: array[qubit, n_w_q],
    uncomp_and_op: UncompAND,
) -> None:
    """Uncompute a little-endian unary cascade from qubit 0 toward a high index.

    This is the opposite traversal to :func:`compute_cascade`: work qubits are
    removed starting at the ``index_qreg[0]`` level and proceeding toward
    ``index_qreg[n_i_q - 1]`` or the intermediate level
    ``next_msb_diff_depth``.

    The targets must hold the computed flags. Cleanup uses the independently
    supplied callback and resets each target. Use ``with dagger`` on
    :func:`compute_cascade` when its AND callback supplies the desired adjoint.

    Args:
        next_msb_diff_depth: The cascade depth at which uncomputation stops.
            Depth 0 uncomputes the full cascade.
        current_cascade_depth: The current deepest computed cascade depth.
        bools: The little-endian boolean values used to compute the work qubits.
        index_qreg: The little-endian index qubit register.
        work_qreg: The work-qubit array, uncomputed from position 0 upward.
        uncomp_and_op: The AND operation used to uncompute each work qubit.

    """
    most_significant_index = len(index_qreg) - 1
    for cascade_depth in range(current_cascade_depth, next_msb_diff_depth, -1):
        work_index = most_significant_index - cascade_depth
        _cascade_step(
            uncomp_and_op,
            most_significant_index,
            work_index,
            bools,
            index_qreg,
            work_qreg,
            False,
        )
        reset(work_qreg[work_index])


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
    CompAND: Unitary[[qubit, qubit, qubit], None],
    UncompAND: Unitary[[qubit, qubit, qubit], None],
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
          step.
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
    index_bools, diff_depths = _get_bools_and_diffs(n_index_qubits, n_index_el)
    _select_with_cascade(
        controlled_ops,
        comp_and_op,
        uncomp_and_op,
        index_qreg,
        state_qreg,
        index_bools,
        diff_depths,
    )


@guppy
@no_type_check
def _select_with_cascade[
    TargetRegs,
    n_index_el: nat,
    n_index_qubits: nat,
    CompAND: Unitary[[qubit, qubit, qubit], None],
    UncompAND: Unitary[[qubit, qubit, qubit], None],
](
    controlled_ops: array[Callable[[qubit, TargetRegs], None], n_index_el],
    comp_and_op: CompAND,
    uncomp_and_op: UncompAND,
    index_qreg: array[qubit, n_index_qubits],
    state_qreg: TargetRegs,
    index_bools: array[array[bool, n_index_qubits], n_index_el],  # ty: ignore[not-subscriptable]
    diff_depths: array[int, n_index_el],
) -> None:
    """Apply SELECT using the supplied address patterns and their differing depths."""
    if n_index_qubits == 1:
        for iteration_index in range(n_index_el):
            if not index_bools[iteration_index][0]:
                x(index_qreg[0])
            controlled_ops[iteration_index](index_qreg[0], state_qreg)
            if not index_bools[iteration_index][0]:
                x(index_qreg[0])
        return

    max_cascade_depth = len(index_qreg) - 1
    work_qreg = qarray(comptime(n_index_qubits - 1))
    for iteration_index in range(n_index_el):
        diff_depth = diff_depths[iteration_index]
        if diff_depth != 0:
            adjacent_and(
                diff_depth, index_qreg, work_qreg, index_bools[iteration_index]
            )

        if diff_depth != max_cascade_depth:
            compute_cascade(
                comp_and_op,
                max_cascade_depth,
                diff_depth,
                index_bools[iteration_index],
                index_qreg,
                work_qreg,
            )

        # The final work qubit is the AND target at the index_qreg[0] level.
        controlled_ops[iteration_index](work_qreg[0], state_qreg)

        next_diff_depth = (
            0 if iteration_index == n_index_el - 1 else diff_depths[iteration_index + 1]
        )
        uncompute_cascade(
            next_diff_depth,
            max_cascade_depth,
            index_bools[iteration_index],
            index_qreg,
            work_qreg,
            uncomp_and_op,
        )

    discard_array(work_qreg)


@guppy.comptime
@no_type_check
def cntrl_select_unary_iteration[
    TargetRegs,
    n_index_qubits: nat,
    n_index_elements: nat,
    CompAND: Unitary[[qubit, qubit, qubit], None],
    UncompAND: Unitary[[qubit, qubit, qubit], None],
](
    controlled_ops: array[Callable[[qubit, TargetRegs], None], n_index_elements],
    comp_and_op: CompAND,
    uncomp_and_op: UncompAND,
    control: qubit,
    index_qreg: array[qubit, n_index_qubits],
    state_qreg: TargetRegs,
) -> None:
    """Apply unary SELECT with an extra, most-significant control bit.

    All visited address patterns include a positive control. Treating it as
    another address qubit shares the ordinary cascade and preserves the
    controlled SELECT's gate and work-qubit counts.
    """
    if n_index_qubits < ceil(log2(n_index_elements)):
        raise ValueError(
            f"{n_index_qubits} qubits is not enough to iterate over "
            f"{n_index_elements} terms"
        )
    index_bools = [
        [*bits, True]
        for bits in get_index_bools(n_index_qubits, n_index_elements, int_to_bits)
    ]
    _select_with_cascade(
        controlled_ops,
        comp_and_op,
        uncomp_and_op,
        [*index_qreg, control],
        state_qreg,
        index_bools,
        msb_diff_depths(index_bools),
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
