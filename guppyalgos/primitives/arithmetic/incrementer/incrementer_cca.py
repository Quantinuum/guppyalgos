"""Incrementer in linear depth using conditionally clean ancillas.

This implements the linear-depth CCA construction described in Section 6.2 of

    Tanuj Khattar, Craig Gidney,
    "Rise of conditionally clean ancillae for efficient quantum circuit constructions",
    https://arxiv.org/abs/2407.17966

This uses 3n + O(1) Toffoli gates (some of these are temporary AND and uncompute gates)
and O(log*(n)) clean ancilla qubits to increment a register of size n.

"""

from __future__ import annotations

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, comptime, nat, panic
from guppylang.std.quantum import cx, discard, discard_array, qubit, toffoli, x
from guppylang.std.mem import mem_swap
from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.utils import qarray
from guppyalgos.utils.guppy.unsafe_borrow import (
    _unsafe_array_borrow,
    _unsafe_array_unborrow,
)
from guppyalgos.utils.guppy.array import join_arrays, split_array


@guppy(unitary=True)
@no_type_check
def cat_cca(q0: qubit, q1: qubit, target: qubit) -> None:
    """Catalyze a conditionally clean ancilla.

    This uses one conditionally clean ancilla (``target``) to catalyze the generation
    of two more conditionally clean ancillae (``q0`` and ``q1``).

    The two operations ``X(target)`` and ``CCX(q0, q1, target)`` commute, making
    this self-inverse.
    """
    x(target)
    toffoli(q0, q1, target)


def _prefix_region_count(n_bits: int) -> int:
    """Return the number of CCA growth regions for ``n_bits`` inputs."""
    if n_bits <= 0:
        return 0

    start = 0
    layer_id = 0
    count = 0
    while start < n_bits:
        start += 2**layer_id + 1
        layer_id += 1
        count += 1
    return count


def _recursive_level_counts(n_qubits: int) -> list[int]:
    """Return the number of prefix inputs at each recursion depth."""
    if n_qubits <= 2:
        return []

    counts = [n_qubits - 1]

    while _prefix_region_count(counts[-1]) > 2:
        counts.append(_prefix_region_count(counts[-1]))

    return counts


def _plan_depths(n_q: int) -> int:
    """Return the number of recursive-level slots in the CCA plan."""
    if n_q <= 2:
        return 0
    return len(_recursive_level_counts(n_q))


def required_clean_ancillas(n_qubits: int) -> int:
    """Return the clean-ancilla count required by the recursive CCA construction."""
    n_levels = _plan_depths(n_qubits)
    return 0 if n_levels == 0 else 2 * n_levels - 1


def _plan_slots(n_q: int) -> int:
    """Return the total number of logical layer slots in the CCA plan.

    slot = depth * layer_stride + layer_idx
    layer_stride = n_q + 2
    """
    return _plan_depths(n_q) * (n_q + 2)


def _plan_level_bits(n_q: int) -> int:
    """Return the total number of input indices stored across all recursion levels."""
    return sum(_recursive_level_counts(n_q))


def _plan_ops_per_layer(n_q: int) -> int:
    """Return the maximum number of operations attached to one logical layer."""
    n_levels = _plan_depths(n_q)
    if n_levels == 0:
        return 0
    return 2 * n_levels - 1


def _plan_ops(n_q: int) -> int:
    """Return the total capacity of the flattened per layer operation buffer."""
    return _plan_slots(n_q) * _plan_ops_per_layer(n_q)


@guppy
@no_type_check
def _x_at[n_q: nat, n_a: nat](
    q: array[qubit, n_q],
    ancilla: array[qubit, n_a],
    target: int,
) -> None:
    """Apply X to a flattened storage/ancilla index.

    Args:
    ----
        q: Storage register
        ancilla: Clean ancilla register
        target: Flattened index of the target

    """
    if target < n_q:
        x(q[target])
    else:
        x(ancilla[target - n_q])


@guppy
@no_type_check
def _cx_at[n_q: nat, n_a: nat](
    q: array[qubit, n_q],
    ancilla: array[qubit, n_a],
    control: int,
    target: int,
) -> None:
    """Apply CX to flattened storage/ancilla indices.

    Args:
    ----
        q: Storage register
        ancilla: Clean ancilla register
        control: Flattened index of the control
        target: Flattened index of the target

    """
    if control < n_q:
        if target < n_q:
            cx(q[control], q[target])
        else:
            cx(q[control], ancilla[target - n_q])
    else:
        if target < n_q:
            cx(ancilla[control - n_q], q[target])
        else:
            cx(ancilla[control - n_q], ancilla[target - n_q])


@guppy
@no_type_check
def _toffoli_at[n_q: nat, n_a: nat](
    q: array[qubit, n_q],
    ancilla: array[qubit, n_a],
    control0: int,
    control1: int,
    target: int,
) -> None:
    """Apply Toffoli to flattened storage/ancilla indices.

    Args:
    ----
        q: Storage register
        ancilla: Clean ancilla register
        control0: Flattened index of the first control
        control1: Flattened index of the second control
        target: Flattened index of the target

    """
    if control0 < n_q:
        if control1 < n_q:
            if target < n_q:
                toffoli(q[control0], q[control1], q[target])
            else:
                toffoli(q[control0], q[control1], ancilla[target - n_q])
        else:
            if target < n_q:
                toffoli(q[control0], ancilla[control1 - n_q], q[target])
            else:
                toffoli(
                    q[control0],
                    ancilla[control1 - n_q],
                    ancilla[target - n_q],
                )
    else:
        if control1 < n_q:
            if target < n_q:
                toffoli(ancilla[control0 - n_q], q[control1], q[target])
            else:
                toffoli(
                    ancilla[control0 - n_q],
                    q[control1],
                    ancilla[target - n_q],
                )
        else:
            if target < n_q:
                toffoli(
                    ancilla[control0 - n_q],
                    ancilla[control1 - n_q],
                    q[target],
                )
            else:
                toffoli(
                    ancilla[control0 - n_q],
                    ancilla[control1 - n_q],
                    ancilla[target - n_q],
                )


@guppy
@no_type_check
def _cat_cca_at[n_q: nat, n_a: nat](
    q: array[qubit, n_q],
    ancilla: array[qubit, n_a],
    q0: int,
    q1: int,
    target: int,
) -> None:
    """Apply ``cat_cca`` using flattened storage/ancilla indices.

    Args:
    ----
        q: Storage register
        ancilla: Clean ancilla register
        q0: Flattened index of the first control
        q1: Flattened index of the second control
        target: Flattened index of the target

    """
    _x_at(q, ancilla, target)
    _toffoli_at(q, ancilla, q0, q1, target)


@guppy
@no_type_check
def _temp_and_compute_at[n_q: nat, n_a: nat](
    q: array[qubit, n_q],
    ancilla: array[qubit, n_a],
    control0: int,
    control1: int,
    target: int,
) -> None:
    """Compute a temporary AND at flattened indices.

    The target is a physical ancilla index.

    Args:
    ----
        q: Storage register
        ancilla: Clean ancilla register
        control0: Flattened index of the first control
        control1: Flattened index of the second control
        target: Flattened index of the target

    """
    ancilla_target = target - n_q

    if control0 < n_q:
        if control1 < n_q:
            temp_and_compute(q[control0], q[control1], ancilla[ancilla_target])
        else:
            temp_and_compute(
                q[control0],
                ancilla[control1 - n_q],
                ancilla[ancilla_target],
            )
    else:
        if control1 < n_q:
            temp_and_compute(
                ancilla[control0 - n_q],
                q[control1],
                ancilla[ancilla_target],
            )
        else:
            temp_and_compute(
                ancilla[control0 - n_q],
                ancilla[control1 - n_q],
                ancilla[ancilla_target],
            )


@guppy
@no_type_check
def _temp_and_uncompute_at[n_q: nat, n_a: nat](
    q: array[qubit, n_q],
    ancilla: array[qubit, n_a],
    control0: int,
    control1: int,
    target: int,
) -> None:
    """Uncompute a temporary AND at flattened indices using MBU.

    The target is a clean ancilla index.

    Args:
    ----
        q: Storage register
        ancilla: Clean ancilla register
        control0: Flattened index of the first control
        control1: Flattened index of the second control
        target: Flattened index of the target

    """
    ancilla_target = target - n_q

    if control0 < n_q:
        if control1 < n_q:
            temp_and_uncompute(q[control0], q[control1], ancilla[ancilla_target])
        else:
            temp_and_uncompute(
                q[control0],
                ancilla[control1 - n_q],
                ancilla[ancilla_target],
            )
    else:
        if control1 < n_q:
            temp_and_uncompute(
                ancilla[control0 - n_q],
                q[control1],
                ancilla[ancilla_target],
            )
        else:
            temp_and_uncompute(
                ancilla[control0 - n_q],
                ancilla[control1 - n_q],
                ancilla[ancilla_target],
            )


@guppy.struct
class _LayerOpPlan[n_slots: nat, n_ops: nat]:
    """Ordered per-layer operations for the fixed-array prefix-AND schedule.

    ``kind`` values are:

        0: Toffoli
        1: conditionally-clean catalysis ``cat_cca``
        2: temporary-AND compute
        3: temporary-AND uncompute

    ``run_forward`` reproduces the filtering in the paper's reference implementation.
    For instance, the odd-indexed ancilla reserved for combining two recursively
    exposed controls is used only during the reverse prefix consumption sweep.

    Args:
    ----
        length: The number of operations belonging to each layer.
        kind: The kind of operation.
        c0: Array of first control indices.
        c1: Array of second control indices.
        target: Array of target indices.
        run_forward: If `True`, the operation is executed during the forward prefix
            generation sweep and inverted during the reverse prefix consumption sweep.
            If `False`, it is skipped during the forward sweep and only its inverse is
            executed during the reverse sweep.

    """

    length: array[int, n_slots]
    kind: array[int, n_ops]
    c0: array[int, n_ops]
    c1: array[int, n_ops]
    target: array[int, n_ops]
    run_forward: array[bool, n_ops]


@guppy.struct
class _CCAPlan[
    n_slots: nat,
    n_depths: nat,
    n_ops: nat,
    n_level_bits: nat,
]:
    """Integer-valued plan for the recursive CCA prefix construction.

    At each recursion level, the incrementer groups input controls into CCA growth
    regions. Each completed region contributes one accumulated condition. If more
    than two accumulated controls remain, those controls become the input bits of
    the next recursion level, where the same reduction procedure is applied again.

    Once a deeper recursion level has reduced its controls, its preparation
    operations are folded back into the parent level. A parent layer may therefore
    acquire additional preparation operations from a deeper recursion level, together
    with a temporary AND compute/uncompute pair when two recursively exposed controls
    must be combined.

    The combine ancilla's temporary-AND operations are marked ``run_forward=False``.
    They are therefore absent from the forward production sweep and become live only
    while the corresponding block is consumed in reverse, exactly like the odd-ancilla
    filter in the Cirq/Qualtran implementation supplied with the paper.

    Args:
    ----
    n_slots: Total number of logical layer slots allocated across all recursion levels.
        Each slot represents one prefix layer and records the one or two controls
        exposing that prefix together with an ordered list of operations attached to
        the layer.

    n_depths: Number of recursive levels for which bookkeeping space is allocated.

    n_ops: Total capacity of the flattened per-layer operation buffer.

    n_level_bits: Total capacity of the packed buffer storing the input bit indices
        for all recursive levels. This is the sum of the input counts of the
        individual recursion levels.

    The plan uses flattened fixed-size layouts.

    Logical layer slots use a fixed stride:

        slot = depth * layer_stride + layer_idx

    where ``layer_stride = n_q + 2``.

    Each logical layer has room for ``ops_per_layer`` attached operations.
    Operation ``i`` belonging to logical layer ``slot`` is stored at

        plan_idx = slot * ops_per_layer + i

    in the flattened arrays contained in ``ops``. These operations include CCA
    catalysis, temporary-AND compute/uncompute operations, and operations spliced
    in from deeper recursion levels.

    The control arrays record, for each logical layer, the one or two controls
    that expose the corresponding prefix condition after recursive folding.
    ``last_control`` retains the local CCA frontier before recursive controls are
    folded into the layer.

    The recursive input indices use a packed layout rather than a fixed stride.
    For recursion depth ``d``, ``level_offset[d]`` gives the start of that level's
    inputs in ``level_bits`` and ``level_count[d]`` gives their number. The
    individual input at local index ``i`` is therefore stored at

        level_bits[level_offset[d] + i]

    for ``0 <= i < level_count[d]``.

    ``level_growth_count`` records the number of CCA growth regions at each
    recursion depth, while ``level_layer_count`` records the corresponding number
    of logical layers.

    """

    # Algorithm dimensions.
    n_q: int
    n_a: int
    layer_stride: int
    ops_per_layer: int

    # Prefix controls for each layer.
    control0: array[int, n_slots]
    control1: array[int, n_slots]
    n_controls: array[int, n_slots]
    last_control: array[int, n_slots]

    # Information associated with each recursive level.
    level_count: array[int, n_depths]
    level_offset: array[int, n_depths]
    level_growth_count: array[int, n_depths]
    level_layer_count: array[int, n_depths]
    level_bits: array[int, n_level_bits]

    ops: _LayerOpPlan[n_slots, n_ops]


def _max_ops_per_layer(n_qubits: int) -> int:
    """Return the maximum number of operations per layer.

    This is a conservation bound: one local operation plus up to two operations
    contributed at every recursion level.
    """
    depth = _plan_depths(n_qubits)
    return 2 * depth + 2


def _check_cca_plan_dimensions(n_qubits: int) -> bool:
    """Validate all CCA planner dimensions."""
    if n_qubits <= 2:
        return True

    levels = _recursive_level_counts(n_qubits)
    n_levels = len(levels)

    assert _plan_slots(n_qubits) == n_levels * (n_qubits + 2)
    assert _plan_level_bits(n_qubits) == sum(levels)

    assert _plan_ops_per_layer(n_qubits) == 2 * n_levels - 1
    assert _plan_ops(n_qubits) == (
        _plan_slots(n_qubits) * _plan_ops_per_layer(n_qubits)
    )

    return True


@guppy
@no_type_check
def _append_layer_op[
    n_slots: nat,
    n_depths: nat,
    n_ops: nat,
    n_level_bits: nat,
](
    plan: _CCAPlan[n_slots, n_depths, n_ops, n_level_bits],
    slot: int,
    kind: int,
    c0: int,
    c1: int,
    target: int,
    run_forward: bool,
) -> None:
    """Append one operation to a logical layer while preserving operation order."""
    i = plan.ops.length[slot]

    plan_idx = slot * plan.ops_per_layer + i
    plan.ops.kind[plan_idx] = kind
    plan.ops.c0[plan_idx] = c0
    plan.ops.c1[plan_idx] = c1
    plan.ops.target[plan_idx] = target
    plan.ops.run_forward[plan_idx] = run_forward
    plan.ops.length[slot] = i + 1


@guppy
@no_type_check
def _copy_layer_ops[
    n_slots: nat,
    n_depths: nat,
    n_ops: nat,
    n_level_bits: nat,
](
    plan: _CCAPlan[n_slots, n_depths, n_ops, n_level_bits],
    source_slot: int,
    destination_slot: int,
) -> None:
    """Append all operations from one slot to another in their original order."""
    i = 0
    while i < plan.ops.length[source_slot]:
        source_idx = source_slot * plan.ops_per_layer + i

        kind = plan.ops.kind[source_idx]
        c0 = plan.ops.c0[source_idx]
        c1 = plan.ops.c1[source_idx]
        target = plan.ops.target[source_idx]
        run_forward = plan.ops.run_forward[source_idx]

        _append_layer_op(
            plan,
            destination_slot,
            kind,
            c0,
            c1,
            target,
            run_forward,
        )
        i += 1


@guppy
@no_type_check
def _build_cca_growth_layers[
    n_slots: nat,
    n_depths: nat,
    n_ops: nat,
    n_level_bits: nat,
](
    plan: _CCAPlan[n_slots, n_depths, n_ops, n_level_bits],
    scratch_size: nat @ comptime,
) -> int:
    """Build every recursion level of the reference prefix-AND decomposition."""
    cca = array(0 for _ in range(scratch_size))
    new_cca = array(0 for _ in range(scratch_size))
    accumulated_controls = array(0 for _ in range(scratch_size))

    n_q = plan.n_q
    layer_stride = plan.layer_stride

    # Step 1: build the CCA growth layers at each recursive level.
    depth = 0
    deepest_level = 0
    building = True

    while building:
        n_bits = plan.level_count[depth]

        # This is inp_anc[0] at this recursion level.
        # Non-final levels reserve the following ancilla (offset 2*depth+1) for
        # recursive control combining.
        seed_offset = 2 * depth
        clean_ancilla = n_q + seed_offset

        cca[0] = clean_ancilla
        cca_len = 1
        accumulated_len = 0

        # Layer zero represents the unconditional toggle at the bottom of the
        # corresponding prefix construction.
        layer_zero = depth * layer_stride
        plan.n_controls[layer_zero] = 0

        growth_layer = 0
        growth_width = 2
        start = 0

        while start < n_bits:
            end = start + growth_width
            if end > n_bits:
                end = n_bits

            # Before generating any new CCAs in the region, the prefix is represented
            # by all previously accumulated controls and the first bit of the new
            # region.  If there are already more than two accumulated controls, only
            # the final local control is needed here; the earlier controls are replaced
            # during Stage 2 by the recursively prepared temporary control.
            layer_idx = start + 1
            slot = depth * layer_stride + layer_idx
            bit_idx = plan.level_bits[plan.level_offset[depth] + start]

            if accumulated_len == 0:
                plan.control0[slot] = bit_idx
                plan.n_controls[slot] = 1
            elif accumulated_len == 1:
                plan.control0[slot] = accumulated_controls[0]
                plan.control1[slot] = bit_idx
                plan.n_controls[slot] = 2
            else:
                # These controls are placeholders until recursive folding below.
                plan.control0[slot] = accumulated_controls[0]
                plan.control1[slot] = accumulated_controls[1]
                plan.n_controls[slot] = accumulated_len + 1
            plan.last_control[slot] = bit_idx

            # Each iteration within a region extends the available prefix by one bit.
            i = start + 1
            while i < end:
                offset = i - start
                q0_idx = plan.level_bits[plan.level_offset[depth] + i]

                if offset == 1:
                    q1_idx = plan.level_bits[plan.level_offset[depth] + i - 1]
                else:
                    q1_idx = cca[cca_len - offset + 1]

                target_idx = cca[cca_len - offset]
                layer_idx = i + 1
                slot = depth * layer_stride + layer_idx

                # Reference code uses And() exactly when t == inp_anc[0].  Every
                # later target is a conditionally-clean borrowed qubit and uses
                # cat_cca.
                if target_idx == clean_ancilla:
                    _append_layer_op(plan, slot, 2, q0_idx, q1_idx, target_idx, True)
                else:
                    _append_layer_op(plan, slot, 1, q0_idx, q1_idx, target_idx, True)

                if accumulated_len == 0:
                    plan.control0[slot] = target_idx
                    plan.n_controls[slot] = 1
                elif accumulated_len == 1:
                    plan.control0[slot] = accumulated_controls[0]
                    plan.control1[slot] = target_idx
                    plan.n_controls[slot] = 2
                else:
                    plan.control0[slot] = accumulated_controls[0]
                    plan.control1[slot] = accumulated_controls[1]
                    plan.n_controls[slot] = accumulated_len + 1
                plan.last_control[slot] = target_idx

                i += 1

            # A one-bit tail performs no CCA, so its accumulated control is simply that
            # input bit.
            if end - start == 1:
                accumulated_controls[accumulated_len] = bit_idx
            else:
                control_index = cca_len + start - end + 1
                accumulated_controls[accumulated_len] = cca[control_index]
            accumulated_len += 1

            retained_start = start - end + 2
            if retained_start < 0:
                retained_start += cca_len

            new_cca_len = 0
            i = retained_start
            while i < cca_len:
                new_cca[new_cca_len] = cca[i]
                new_cca_len += 1
                i += 1

            i = start
            while i < end:
                new_cca[new_cca_len] = plan.level_bits[plan.level_offset[depth] + i]
                new_cca_len += 1
                i += 1

            i = 0
            while i < new_cca_len:
                cca[i] = new_cca[i]
                i += 1

            cca_len = new_cca_len
            growth_layer += 1
            growth_width = 2 * growth_width - 1
            start = end

        plan.level_growth_count[depth] = growth_layer

        # If the completed growth regions leave at most two accumulated controls,
        # the current recursion level is finished.  Otherwise, the accumulated
        # controls become the input bits of the next recursive level.
        if accumulated_len <= 2:
            plan.level_layer_count[depth] = n_bits + 1
            deepest_level = depth
            building = False
        else:
            plan.level_layer_count[depth] = n_bits + 2

            child = depth + 1

            plan.level_count[child] = accumulated_len
            plan.level_offset[child] = (
                plan.level_offset[depth] + plan.level_count[depth]
            )

            i = 0
            while i < accumulated_len:
                plan.level_bits[plan.level_offset[child] + i] = accumulated_controls[i]
                i += 1

            depth += 1

    return deepest_level


@guppy
@no_type_check
def _fold_recursive_cntrl_preparation[
    n_slots: nat,
    n_depths: nat,
    n_ops: nat,
    n_level_bits: nat,
](
    deepest_level: int,
    plan: _CCAPlan[n_slots, n_depths, n_ops, n_level_bits],
) -> None:
    """Fold deeper recursive control preparation back into the parent prefix schedule.

    Each recursion level produces a smaller CCA construction that reduces the
    accumulated controls from the parent level to one or two exposed controls.
    These child schedules are folded back into the parent by splicing the child's
    ordered operations into the first parent layer that needs the corresponding
    reduced prefix condition.

    Folding proceeds from the deepest recursion level upwards:

        level 2 -> level 1
        level 1 -> level 0

    For each parent growth region:

    - the operations attached to the corresponding child prefix layer are appended
    to the first parent slot in the region
    - if the child exposes one control, that control is used directly
    - if the child exposes two controls, they are combined into the parent's
    reserved combine ancilla using a temporary AND
    - every prefix in the parent region is rewritten to use the resulting recursive
    control together with its local CCA frontier
    - when a temporary combine ancilla is used, the corresponding uncompute
    operation is attached to the slot immediately after the region.

    The temporary-AND compute/uncompute operations for the combine ancilla are
    marked ``run_forward=False``. They are therefore skipped during the forward
    prefix-production sweep and only take effect, via their inverses, during the
    reverse prefix-consumption sweep. This reproduces the odd-ancilla filtering in
    the reference Cirq/Qualtran implementation.

    After folding, the recursive prefix schedule is represented as a single flat
    per-layer operation schedule and can be executed without runtime recursion.

    """
    n_q = plan.n_q
    layer_stride = plan.layer_stride

    depth = deepest_level
    while depth > 0:
        child = depth
        parent = depth - 1
        n_bits = plan.level_count[parent]
        n_growth_layers = plan.level_growth_count[parent]

        # Growth layer zero does not need recursive preparation. The first outer
        # layer that needs the reduced control starts at prefix position 2.
        layer_id = 1
        start = 2
        growth_width = 3

        while layer_id < n_growth_layers:
            end = start + growth_width
            if end > n_bits:
                end = n_bits

            child_slot = child * layer_stride + layer_id
            parent_start_slot = parent * layer_stride + start + 1

            _copy_layer_ops(plan, child_slot, parent_start_slot)

            reduced_n_controls = plan.n_controls[child_slot]

            child_control0 = plan.control0[child_slot]
            child_control1 = plan.control1[child_slot]

            temp_target = 0

            if reduced_n_controls == 1:
                temp_target = child_control0
            elif reduced_n_controls == 2:
                combine_offset = 2 * parent + 1
                if combine_offset >= plan.n_a:
                    panic("missing reserved recursive-control combine ancilla")
                temp_target = n_q + combine_offset

                # The paper adds And(child0, child1, inp_anc[1]) at block start,
                # but Incrementer omits odd-ancilla-targeted ops from the forward
                # sweep.

                _append_layer_op(
                    plan,
                    parent_start_slot,
                    2,
                    child_control0,
                    child_control1,
                    temp_target,
                    False,
                )
            else:
                panic("malformed recursive control schedule")

            i = start
            while i < end:
                slot = parent * layer_stride + i + 1
                local_frontier = plan.last_control[slot]
                plan.control0[slot] = temp_target
                plan.control1[slot] = local_frontier
                plan.n_controls[slot] = 2
                i += 1

            if reduced_n_controls == 2:
                parent_end_slot = parent * layer_stride + end + 1
                child_control0 = plan.control0[child_slot]
                child_control1 = plan.control1[child_slot]

                _append_layer_op(
                    plan,
                    parent_end_slot,
                    3,
                    child_control0,
                    child_control1,
                    temp_target,
                    False,
                )

            layer_id += 1
            start = end
            growth_width = 2 * growth_width - 1

        depth -= 1


@guppy
@no_type_check
def _apply_layer_op[n_q: nat, n_a: nat](
    q: array[qubit, n_q],
    ancilla: array[qubit, n_a],
    kind: int,
    c0: int,
    c1: int,
    target: int,
) -> None:
    """Execute one planned operation."""
    if kind == 0:
        _toffoli_at(q, ancilla, c0, c1, target)
    elif kind == 1:
        _cat_cca_at(q, ancilla, c0, c1, target)
    elif kind == 2:
        _temp_and_compute_at(q, ancilla, c0, c1, target)
    elif kind == 3:
        _temp_and_uncompute_at(q, ancilla, c0, c1, target)
    else:
        panic("unknown CCA planner operation")


@guppy
@no_type_check
def _apply_inverse_layer_op[n_q: nat, n_a: nat](
    q: array[qubit, n_q],
    ancilla: array[qubit, n_a],
    kind: int,
    c0: int,
    c1: int,
    target: int,
) -> None:
    """Execute the inverse/adjoint operation used on the reverse sweep."""
    if kind == 0:
        _toffoli_at(q, ancilla, c0, c1, target)
    elif kind == 1:
        _cat_cca_at(q, ancilla, c0, c1, target)
    elif kind == 2:
        _temp_and_uncompute_at(q, ancilla, c0, c1, target)
    elif kind == 3:
        _temp_and_compute_at(q, ancilla, c0, c1, target)
    else:
        panic("unknown CCA planner operation")


@guppy
@no_type_check
def _apply_cca_plan[
    n_q: nat,
    n_a: nat,
    n_slots: nat,
    n_depths: nat,
    n_ops: nat,
    n_level_bits: nat,
](
    q: array[qubit, n_q],
    ancilla: array[qubit, n_a],
    plan: _CCAPlan[n_slots, n_depths, n_ops, n_level_bits],
) -> None:
    """Produce all prefix states, then consume them in reverse."""
    n_layers = plan.level_layer_count[0]

    # Forward production sweep.
    layer = 0
    while layer < n_layers:
        i = 0
        while i < plan.ops.length[layer]:
            plan_idx = layer * plan.ops_per_layer + i
            if plan.ops.run_forward[plan_idx]:
                _apply_layer_op(
                    q,
                    ancilla,
                    plan.ops.kind[plan_idx],
                    plan.ops.c0[plan_idx],
                    plan.ops.c1[plan_idx],
                    plan.ops.target[plan_idx],
                )
            i += 1
        layer += 1

    # Reverse consumption sweep.
    layer = n_layers
    while layer > 0:
        layer -= 1

        if layer < int(n_q):
            n_controls = plan.n_controls[layer]
            if n_controls == 0:
                _x_at(q, ancilla, layer)
            elif n_controls == 1:
                _cx_at(q, ancilla, plan.control0[layer], layer)
            elif n_controls == 2:
                _toffoli_at(
                    q,
                    ancilla,
                    plan.control0[layer],
                    plan.control1[layer],
                    layer,
                )
            else:
                panic("CCA schedule produced more than two final controls")

        i = plan.ops.length[layer]
        while i > 0:
            i -= 1
            plan_idx = layer * plan.ops_per_layer + i
            _apply_inverse_layer_op(
                q,
                ancilla,
                plan.ops.kind[plan_idx],
                plan.ops.c0[plan_idx],
                plan.ops.c1[plan_idx],
                plan.ops.target[plan_idx],
            )


@guppy
@no_type_check
def _empty_cca_plan(
    n_q: nat @ comptime,
    n_a: nat @ comptime,
    n_slots: nat @ comptime,
    n_depths: nat @ comptime,
    n_ops_per_layer: nat @ comptime,
    n_ops: nat @ comptime,
    n_level_bits: nat @ comptime,
) -> _CCAPlan["n_slots", "n_depths", "n_ops", "n_level_bits"]:  # noqa: UP037, F821
    """Allocate an empty CCA schedule plan."""
    ops = _LayerOpPlan[n_slots, n_ops](
        array(0 for _ in range(n_slots)),
        array(0 for _ in range(n_ops)),
        array(0 for _ in range(n_ops)),
        array(0 for _ in range(n_ops)),
        array(0 for _ in range(n_ops)),
        array(False for _ in range(n_ops)),
    )

    return _CCAPlan[n_slots, n_depths, n_ops, n_level_bits](
        int(n_q),
        int(n_a),
        int(n_q) + 2,
        int(n_ops_per_layer),
        array(0 for _ in range(n_slots)),
        array(0 for _ in range(n_slots)),
        array(0 for _ in range(n_slots)),
        array(0 for _ in range(n_slots)),
        array(0 for _ in range(n_depths)),
        array(0 for _ in range(n_depths)),
        array(0 for _ in range(n_depths)),
        array(0 for _ in range(n_depths)),
        array(0 for _ in range(n_level_bits)),
        ops,
    )


@guppy
@no_type_check
def _cca_incrementer_with_ancillas[
    n_q: nat,
    n_a: nat,
    n_slots: nat,
    n_depths: nat,
    n_ops: nat,
    n_level_bits: nat,
](
    q: array[qubit, n_q],
    ancilla: array[qubit, n_a],
    plan: _CCAPlan[n_slots, n_depths, n_ops, n_level_bits],
) -> None:
    r"""Apply the recursively reduced CCA incrementer using explicit ancillas.

    The implementation has three stages:

        1. Build the CCA growth layers at each recursive control-reduction level.
        2. Fold the deeper control-reduction levels back into their parent layers.
        3. Apply the complete forward schedule, then toggle each output bit while
           reversing the schedule.

    The implementation uses a flattened ``nat`` index space:

        storage:       0, ..., n_q - 1
        clean ancilla: n_q, ..., n_q + n_a - 1

    Non-final recursive levels reserve a pair of clean ancillas exactly: one CCA seed
    and one reverse sweep temporary for combining two recursively exposed controls.
    The child recursion receives the remaining ancillas after that pair.
    The deepest level needs only its CCA seed.

    """
    # The top recursion level acts on all storage bits except the most significant
    # output bit.
    plan.level_count[0] = int(n_q) - 1
    plan.level_offset[0] = 0

    i = 0
    while i < int(n_q) - 1:
        plan.level_bits[i] = i
        i += 1

    deepest_level = _build_cca_growth_layers(plan, comptime(n_q))
    _fold_recursive_cntrl_preparation(deepest_level, plan)
    _apply_cca_plan(q, ancilla, plan)


@guppy
@no_type_check
def cca_incrementer[n_q: nat](
    q: array[qubit, n_q],
) -> None:
    """Increment ``q`` using the required number of clean CCA ancillas."""
    if comptime(n_q == 1):
        x(q[0])
        return

    if comptime(n_q == 2):
        cx(q[0], q[1])
        x(q[0])
        return

    comptime(_check_cca_plan_dimensions(n_q))

    ancilla = qarray(comptime(required_clean_ancillas(n_q)))

    plan = _empty_cca_plan(
        comptime(n_q),
        comptime(required_clean_ancillas(n_q)),
        comptime(_plan_slots(n_q)),
        comptime(_plan_depths(n_q)),
        comptime(_plan_ops_per_layer(n_q)),
        comptime(_plan_ops(n_q)),
        comptime(_plan_level_bits(n_q)),
    )

    _cca_incrementer_with_ancillas(q, ancilla, plan)
    discard_array(ancilla)


@guppy
@no_type_check
def _apply_cca_plan_dagger[
    n_q: nat,
    n_a: nat,
    n_slots: nat,
    n_depths: nat,
    n_ops: nat,
    n_level_bits: nat,
](
    q: array[qubit, n_q],
    ancilla: array[qubit, n_a],
    plan: _CCAPlan[n_slots, n_depths, n_ops, n_level_bits],
) -> None:
    """Apply the inverse logical action of the CCA incrementer plan."""
    n_layers = plan.level_layer_count[0]

    layer = 0
    while layer < n_layers:
        i = 0
        while i < plan.ops.length[layer]:
            plan_idx = layer * plan.ops_per_layer + i
            _apply_layer_op(
                q,
                ancilla,
                plan.ops.kind[plan_idx],
                plan.ops.c0[plan_idx],
                plan.ops.c1[plan_idx],
                plan.ops.target[plan_idx],
            )
            i += 1

        if layer < int(n_q):
            n_controls = plan.n_controls[layer]
            if n_controls == 0:
                _x_at(q, ancilla, layer)
            elif n_controls == 1:
                _cx_at(q, ancilla, plan.control0[layer], layer)
            elif n_controls == 2:
                _toffoli_at(
                    q,
                    ancilla,
                    plan.control0[layer],
                    plan.control1[layer],
                    layer,
                )
            else:
                panic("CCA schedule produced more than two final controls")

        layer += 1

    layer = n_layers
    while layer > 0:
        layer -= 1
        i = plan.ops.length[layer]
        while i > 0:
            i -= 1
            plan_idx = layer * plan.ops_per_layer + i
            if plan.ops.run_forward[plan_idx]:
                _apply_inverse_layer_op(
                    q,
                    ancilla,
                    plan.ops.kind[plan_idx],
                    plan.ops.c0[plan_idx],
                    plan.ops.c1[plan_idx],
                    plan.ops.target[plan_idx],
                )


@guppy
@no_type_check
def cca_decrementer[n_q: nat](
    q: array[qubit, n_q],
) -> None:
    """Decrement ``q`` using the inverse CCA incrementer construction."""
    if comptime(n_q == 1):
        x(q[0])
        return

    if comptime(n_q == 2):
        x(q[0])
        cx(q[0], q[1])
        return

    ancilla = qarray(comptime(required_clean_ancillas(n_q)))

    plan = _empty_cca_plan(
        comptime(n_q),
        comptime(required_clean_ancillas(n_q)),
        comptime(_plan_slots(n_q)),
        comptime(_plan_depths(n_q)),
        comptime(_plan_ops_per_layer(n_q)),
        comptime(_plan_ops(n_q)),
        comptime(_plan_level_bits(n_q)),
    )

    plan.level_count[0] = int(n_q) - 1
    plan.level_offset[0] = 0

    i = 0
    while i < int(n_q) - 1:
        plan.level_bits[i] = i
        i += 1

    deepest_level = _build_cca_growth_layers(plan, comptime(n_q))
    _fold_recursive_cntrl_preparation(deepest_level, plan)

    _apply_cca_plan_dagger(q, ancilla, plan)

    discard_array(ancilla)


# TODO: this is a general method for controlling any incrementer
@guppy
@no_type_check
def cntrl_cca_incrementer[n_q: nat](
    control: qubit,
    q: array[qubit, n_q],
) -> None:
    """Call the controlled CCA-based incrementer.

    This constructs a controlled incrementer by treating the `n_c` control qubits as
    the least-significant qubits of an `(n_c + n_q)`-qubit incrementer, followed by an
    `n_c`-qubit decrementer acting on the controls.

    A multiply controlled incrementer can therefore be constructed recursively by
    adding one control qubit at a time.

    For example, starting from an `n`-qubit incrementer:

    ```
    Inc_n(q_{n-1} ... q_0)
    ```

    adding one control `c_0` gives

    ```
    Inc_{n+1}(c_0 q_{n-1} ... q_0) Dec_1(c_0).
    ```

    Adding a second control `c_1` recursively gives

    ```
    Inc_{n+2}(c_1 c_0 q_{n-1} ... q_0)
    Dec_1(c_1)
    Dec_2(c_1 c_0)
    Inc_1(c_1).
    ```

    Since `Dec_1(c_1)` and `Inc_1(c_1)` cancel, we are left with the expected doubly
    controlled construction:

    ```
    Inc_{n+2}(c_1 c_0 q_{n-1} ... q_0) Dec_2(c_1 c_0).
    ```

    `Dec_1(c_1)` and `Inc_1(c_1)` are single-qubit X gates. More generally, each
    additional control introduces a canceling pair of single-qubit X gates at the
    corresponding recursion level. A compiler can remove these pairs provided it
    optimizes the circuit at each recursive level where the cancellation is exposed.
    """
    if comptime(n_q == 1):
        cx(control, q[0])
        return

    if comptime(n_q == 2):
        toffoli(control, q[0], q[1])
        cx(control, q[0])
        return

    # TODO: this uses one extra ancilla than necessary
    control_owned = qubit()
    mem_swap(control_owned, control)
    controls = array(control_owned)

    controls_borrowed = _unsafe_array_borrow(controls)
    q_borrowed = _unsafe_array_borrow(q)

    flat = join_arrays(controls_borrowed, q_borrowed, comptime(n_q + 1))

    cca_incrementer(flat)

    controls_borrowed, q_borrowed = split_array(flat, 1, n_q)

    _unsafe_array_unborrow(controls, controls_borrowed)
    _unsafe_array_unborrow(q, q_borrowed)

    cca_decrementer(controls)

    control_owned = controls.take(0)
    controls.discard_all_taken()

    mem_swap(control_owned, control)
    discard(control_owned)


# TODO: this is a general method for controlling any decrementer
@guppy
@no_type_check
def cntrl_cca_decrementer[n_q: nat](
    control: qubit,
    q: array[qubit, n_q],
) -> None:
    """Call the controlled CCA-based decrementer.

    This is a general method for returning a controlled decrementer. The ``n_c`` control
    qubits become the least significant qubits of a (n_c + n_q)-qubit decrementer.

    It should be possible to define this recursively, i.e. with one control.
    """
    if comptime(n_q == 1):
        cx(control, q[0])
        return

    if comptime(n_q == 2):
        cx(control, q[0])
        toffoli(control, q[0], q[1])
        return

    # TODO: this uses one extra ancilla than necessary
    control_owned = qubit()
    mem_swap(control_owned, control)
    controls = array(control_owned)

    cca_incrementer(controls)

    controls_borrowed = _unsafe_array_borrow(controls)
    q_borrowed = _unsafe_array_borrow(q)

    flat = join_arrays(controls_borrowed, q_borrowed, comptime(n_q + 1))

    cca_decrementer(flat)

    controls_borrowed, q_borrowed = split_array(flat, 1, n_q)

    _unsafe_array_unborrow(controls, controls_borrowed)
    _unsafe_array_unborrow(q, q_borrowed)

    control_owned = controls.take(0)
    controls.discard_all_taken()

    mem_swap(control_owned, control)
    discard(control_owned)
