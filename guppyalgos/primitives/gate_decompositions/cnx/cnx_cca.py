"""Multi-controlled X gate in log depth using conditionally clean ancillas.

This implements the logarithmic-depth CCA construction described in Section 5.3 of

    Tanuj Khattar, Craig Gidney,
    "Rise of conditionally clean ancillae for efficient quantum circuit constructions",
    https://arxiv.org/abs/2407.17966

This uses 2 * n_controls - 3 Toffoli gates (some of these are temporary AND and
uncompute gates) and either 1 or 2 clean ancilla qubits,
depending on the number of controls.

The dirty variant (Section 5.5) uses two dirty ancillae and 4n - 6 Toffoli gates.
"""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.quantum import qubit, toffoli, x, cx, discard
from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_compute,
    temp_and_uncompute,
)


@guppy
@no_type_check
def _and_compute(q0: qubit, q1: qubit, target: qubit, dirty: bool) -> None:
    """Compute an AND, with a Toffoli gate when the target may be dirty."""
    if dirty:
        toffoli(q0, q1, target)
    else:
        temp_and_compute(q0, q1, target)


@guppy
@no_type_check
def _and_uncompute(q0: qubit, q1: qubit, target: qubit, dirty: bool) -> None:
    """Uncompute an AND, with a Toffoli gate when the target may be dirty."""
    if dirty:
        toffoli(q0, q1, target)
    else:
        temp_and_uncompute(q0, q1, target)


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


@guppy
@no_type_check
def _apply_cat_cca[n_controls: nat](
    control: array[qubit, n_controls],
    allocated_ancilla: qubit,
    control0: int,
    control1: int,
    gate_target: int,
) -> None:
    """Apply one planned CCA operation."""
    allocated_idx = n_controls

    if gate_target == allocated_idx:
        cat_cca(control[control0], control[control1], allocated_ancilla)
    else:
        if control0 == allocated_idx:
            cat_cca(allocated_ancilla, control[control1], control[gate_target])
        elif control1 == allocated_idx:
            cat_cca(control[control0], allocated_ancilla, control[gate_target])
        else:
            cat_cca(control[control0], control[control1], control[gate_target])


@guppy
@no_type_check
def _log_ladder_one_ancilla[n_controls: nat](
    control: array[qubit, n_controls],
    target: qubit,
    allocated_ancilla: qubit,
    dirty: bool,
) -> None:
    """Apply the complete CCA construction for three to five controls.

    With ``dirty`` set, the final uncompute is skipped: applying the
    function twice then implements the ladder using a dirty ancilla.
    """
    _and_compute(control[0], control[1], allocated_ancilla, dirty)
    if n_controls == 3:
        toffoli(allocated_ancilla, control[2], target)
    elif n_controls == 4:
        cat_cca(control[2], control[3], control[1])
        toffoli(allocated_ancilla, control[1], target)
        cat_cca(control[2], control[3], control[1])
    else:
        cat_cca(control[3], control[4], control[1])
        cat_cca(control[1], control[2], control[0])
        toffoli(allocated_ancilla, control[0], target)
        cat_cca(control[1], control[2], control[0])
        cat_cca(control[3], control[4], control[1])
    if not dirty:
        temp_and_uncompute(control[0], control[1], allocated_ancilla)


@guppy
@no_type_check
def _log_ladder_two_ancillas[n_controls: nat](
    control: array[qubit, n_controls],
    target: qubit,
    allocated_ancilla: qubit,
    action_ancilla: qubit,
    dirty: bool,
) -> None:
    r"""Apply the two-ancilla logarithmic CCA construction.

    With ``dirty`` set, the ancillas may be dirty.

    The implementation follows the logarithmic-depth construction of Khattar and
    Gidney. It first builds an integer-valued plan in Guppy arrays and then applies
    the corresponding operations.

    The construction has two stages:

        1. A logarithmic CCA-growth stage reduces the original set of controls to
        $O(\log n)$ remaining conditions.
        2. A linear CCA ladder acts on those remaining conditions using the second
        clean ancilla, applies the final Toffoli to ``target``, and is then
        uncomputed.

    At the start of round i, the controls are partitioned as

        [ conditions | CCAs | unused controls ]
              i        2**i

    where the first i wires (together with the allocated clean ancilla)
    encode the accumulated conditions, the next 2**i wires are
    conditionally clean ancillas, and the remaining controls have not yet
    been incorporated.

    Each outer round doubles the number of conditionally clean ancillas by
    incorporating the next block of controls. This is planned by the inner
    loop, which performs $i + 1$ narrowing substeps for the transition
    $i -> i + 1$. Within each substep, the generated Toffoli operations act on
    pairwise-disjoint triples and may therefore be executed in parallel.

    The main index buffers correspond to the conceptual partition as follows:

    ``remaining``
        The unprocessed suffix of control indices.

    ``cca``
        The current set of available conditionally clean targets. Only the live
        prefix of this buffer, of length ``cca_len``, is valid.

    ``final_controls``
        The residual $O(\log n)$ conditions on which the middle linear CCA
        ladder acts after the outer growth stage has completed.

    During one narrowing substep, the live prefix of ``cca`` represents the
    paper's workspace set $P_j$, while the live prefix of ``new_cca``
    represents the current unmarked set $Q_j$.

    For each batch, the code records operations of the form

    .. math::

        (t_k, x_k, y_k),

    where $t_k$ is drawn from $P_j$ and $x_k, y_k$ are drawn
    from $Q_j$. In the code, these are stored as

        outer_c0[k] = x_k
        outer_c1[k] = y_k
        outer_t[k]  = t_k

    After applying such an operation:

    - ``new_sub_cca`` stores the newly marked inputs $x_k$ and $y_k$;
    - ``next_new_cca`` stores the surviving element of $Q_{j+1}$ together
    with the used targets $t_k$;
    - ``cca_len`` is reduced to remove targets that are no longer available as
    workspace.

    The implementation uses a flattened ``nat`` index space:

        controls:          0, ..., n_controls - 1
        allocated ancilla: n_controls
        action ancilla:    n_controls + 1

    The CnX target is passed separately and is not stored in these index buffers.

    After the outer logarithmic ladder is applied, the code constructs and applies
    the middle linear CCA ladder on

        [action_ancilla, *final_controls]

    It then applies the final Toffoli to ``target``, reverses the linear ladder, and
    finally reverses the outer logarithmic ladder so that all borrowed controls and
    clean ancillas are restored.

    The local loop variables named ``i`` and ``j`` are implementation counters and
    do not coincide with the round and substep indices $i$ and $j$ used in the paper.

    """
    allocated_idx = int(n_controls)
    action_idx = n_controls + 1

    # Unprocessed suffix of controls
    remaining = array(0 for _ in range(comptime(n_controls)))

    i = 0
    while i < int(n_controls):
        remaining[i] = i
        i += 1

    remaining_len = int(n_controls)

    # Current set of available conditionally clean targets
    cca = array(0 for _ in range(comptime(n_controls + 1)))
    cca[0] = allocated_idx
    cca_len = 1

    new_cca = array(0 for _ in range(comptime(n_controls + 1)))

    # Controls and targets in the substeps
    next_new_cca = array(0 for _ in range(comptime(n_controls + 1)))
    new_sub_cca = array(0 for _ in range(comptime(n_controls + 1)))

    outer_c0 = array(0 for _ in range(comptime(n_controls)))
    outer_c1 = array(0 for _ in range(comptime(n_controls)))
    outer_t = array(0 for _ in range(comptime(n_controls)))
    outer_len = 0

    # Collects the ~log n conditions that remain after the logarithmic growth phase
    final_controls = array(0 for _ in range(comptime(n_controls + 1)))
    final_len = 0

    # The outer loop performs O(log n) CCA-growth rounds.
    while remaining_len > 1:
        # At round $i$, there are $2^i$ marked elements.
        # Convert the next $2^i+1$ elements to yield $2^{i+1}$ CCAs.
        n_new_cca = cca_len + 1

        if remaining_len < n_new_cca:
            n_new_cca = remaining_len

        # Add the n_new_cca entries to new_cca.
        i = 0
        while i < n_new_cca:
            new_cca[i] = remaining[i]
            i += 1

        new_cca_len = n_new_cca

        # Remove the first n_new_cca entries from remaining.
        i = n_new_cca
        while i < remaining_len:
            remaining[i - n_new_cca] = remaining[i]
            i += 1

        remaining_len -= n_new_cca
        new_sub_len = 0

        while new_cca_len > 1:
            # $2^{i-j-1}$ parallel operations applied in substep j
            ccx_n = new_cca_len // 2
            odd = new_cca_len % 2

            i = 0
            while i < ccx_n:
                # $x_k = q_{2k + 1}$
                # $y_k = q_{2k + 2}$
                # $t_k = p_{2^{i-j} - 1 - k}$
                outer_c0[outer_len] = new_cca[odd + i]
                outer_c1[outer_len] = new_cca[odd + ccx_n + i]
                outer_t[outer_len] = cca[cca_len - ccx_n + i]

                outer_len += 1
                i += 1

            # The selected $x_k$, $y_k$ input controls become part of the
            # newly available CCA set
            i = odd
            while i < new_cca_len:
                new_sub_cca[new_sub_len] = new_cca[i]
                new_sub_len += 1
                i += 1

            # The targets used from cca are moved into next_new_cca ($Q_{j+1}$)
            i = 0
            while i < ccx_n:
                next_new_cca[i] = cca[cca_len - ccx_n + i]
                i += 1

            # If the number of controls is not a power of two, ``new_cca_len`` is odd,
            # and the element at index 0 is not paired in that substep
            if odd == 1:
                next_new_cca[ccx_n] = new_cca[0]

            cca_len -= ccx_n
            new_cca_len = ccx_n + odd

            i = 0
            while i < new_cca_len:
                new_cca[i] = next_new_cca[i]
                i += 1

        # Newly marked controls are added back to the available CCA pool
        i = 0
        while i < new_sub_len:
            cca[cca_len] = new_sub_cca[i]
            cca_len += 1
            i += 1

        # Sort the live CCA prefix:
        # allocated_idx first, then control indices in ascending order.
        i = 1
        while i < cca_len:
            value = cca[i]
            j = i

            while j > 0:
                previous = cca[j - 1]

                should_shift = False

                if value == allocated_idx:
                    # The allocated ancilla must be first.
                    if previous != allocated_idx:
                        should_shift = True
                elif previous != allocated_idx and previous > value:
                    should_shift = True

                if not should_shift:
                    break

                cca[j] = previous
                j -= 1

            cca[j] = value
            i += 1

        if not (new_cca_len == 1 and new_cca[0] == allocated_idx):
            i = 0
            while i < new_cca_len:
                final_controls[final_len] = new_cca[i]
                final_len += 1
                i += 1

    i = 0
    while i < remaining_len:
        final_controls[final_len] = remaining[i]
        final_len += 1
        i += 1

    final_controls[final_len] = allocated_idx
    final_len += 1

    # Sort final controls using the same ordering.
    i = 1
    while i < final_len:
        value = final_controls[i]
        j = i

        while j > 0:
            previous = final_controls[j - 1]

            should_shift = False

            if value == allocated_idx:
                if previous != allocated_idx:
                    should_shift = True
            elif previous != allocated_idx and previous > value:
                should_shift = True

            if not should_shift:
                break

            final_controls[j] = previous
            j -= 1

        final_controls[j] = value
        i += 1

    # Apply the outer logarithmic ladder.
    _and_compute(
        control[outer_c0[0]],
        control[outer_c1[0]],
        allocated_ancilla,
        dirty,
    )

    if final_len < 5:
        action_control1 = final_controls[5 - final_len]
    else:
        action_control1 = final_controls[0]
        if dirty:
            toffoli(action_ancilla, allocated_ancilla, target)

    i = 1
    while i < outer_len:
        _apply_cat_cca(
            control,
            allocated_ancilla,
            outer_c0[i],
            outer_c1[i],
            outer_t[i],
        )
        i += 1

    # Construct the middle linear ladder on
    #
    #     [action_ancilla, *final_controls]
    linear_c0 = array(0 for _ in range(comptime(n_controls + 1)))
    linear_c1 = array(0 for _ in range(comptime(n_controls + 1)))
    linear_t = array(0 for _ in range(comptime(n_controls + 1)))

    first_c0 = linear_c0[0]
    first_c1 = linear_c1[0]

    linear_len = 0
    ladder_len = final_len + 1

    # Construct the upward part of the linear ladder.
    i = 0
    while i < ladder_len - 2:
        if i == 0:
            linear_c0[linear_len] = final_controls[0]
            linear_c1[linear_len] = final_controls[1]
            linear_t[linear_len] = action_idx
        else:
            linear_c0[linear_len] = final_controls[i]
            linear_c1[linear_len] = final_controls[i + 1]
            linear_t[linear_len] = final_controls[i - 1]

        linear_len += 1
        i += 2

    # Construct the downward part of the linear ladder.
    x_idx = 0
    y_idx = 0
    t_idx = 0

    if ladder_len % 2 == 1:
        if ladder_len > 6:
            x_idx = ladder_len - 3
            y_idx = ladder_len - 5
            t_idx = ladder_len - 6

    else:
        if ladder_len > 5:
            x_idx = ladder_len - 1
            y_idx = ladder_len - 4
            t_idx = ladder_len - 5

    if t_idx > 0:
        x_wire = action_idx
        y_wire = action_idx
        t_wire = action_idx

        if x_idx != 0:
            x_wire = final_controls[x_idx - 1]

        if y_idx != 0:
            y_wire = final_controls[y_idx - 1]

        if t_idx != 0:
            t_wire = final_controls[t_idx - 1]

        linear_c0[linear_len] = x_wire
        linear_c1[linear_len] = y_wire
        linear_t[linear_len] = t_wire
        linear_len += 1

    i = t_idx
    while i > 2:
        linear_c0[linear_len] = final_controls[i - 1]
        linear_c1[linear_len] = final_controls[i - 2]
        linear_t[linear_len] = final_controls[i - 3]

        linear_len += 1
        i -= 2

    # The first linear operation targets the genuinely clean action ancilla.
    first_c0 = linear_c0[0]
    first_c1 = linear_c1[0]

    if first_c0 == allocated_idx:
        _and_compute(
            allocated_ancilla,
            control[first_c1],
            action_ancilla,
            dirty,
        )
    elif first_c1 == allocated_idx:
        _and_compute(
            control[first_c0],
            allocated_ancilla,
            action_ancilla,
            dirty,
        )
    else:
        _and_compute(
            control[first_c0],
            control[first_c1],
            action_ancilla,
            dirty,
        )

    i = 1
    while i < linear_len:
        _apply_cat_cca(
            control,
            allocated_ancilla,
            linear_c0[i],
            linear_c1[i],
            linear_t[i],
        )
        i += 1

    # Apply the middle action Toffoli.
    if action_control1 == allocated_idx:
        toffoli(
            action_ancilla,
            allocated_ancilla,
            target,
        )
    else:
        toffoli(
            action_ancilla,
            control[action_control1],
            target,
        )

    # Reverse the middle linear ladder.
    i = linear_len
    while i > 1:
        i -= 1

        _apply_cat_cca(
            control,
            allocated_ancilla,
            linear_c0[i],
            linear_c1[i],
            linear_t[i],
        )

    if first_c0 == allocated_idx:
        _and_uncompute(
            allocated_ancilla,
            control[first_c1],
            action_ancilla,
            dirty,
        )
    elif first_c1 == allocated_idx:
        _and_uncompute(
            control[first_c0],
            allocated_ancilla,
            action_ancilla,
            dirty,
        )
    else:
        _and_uncompute(
            control[first_c0],
            control[first_c1],
            action_ancilla,
            dirty,
        )

    # Reverse the outer logarithmic ladder.
    i = outer_len
    while i > 1:
        i -= 1

        _apply_cat_cca(
            control,
            allocated_ancilla,
            outer_c0[i],
            outer_c1[i],
            outer_t[i],
        )

    if not dirty:
        temp_and_uncompute(
            control[outer_c0[0]],
            control[outer_c1[0]],
            allocated_ancilla,
        )


@guppy
@no_type_check
def cnx_cca_logdepth[n_controls: nat](
    controls: array[qubit, n_controls],
    target: qubit,
) -> None:
    r"""Apply a logarithmic-depth CCA-based cnx.

    Based on https://arxiv.org/abs/2407.17966, given $n$ control qubits and the
    target, uses 1 or 2 clean ancilla qubits to apply the multicontrolled-x
    gate on the target using $2n -3$ Toffoli gates.

    Args:
        controls (array[qubit, n_controls]): register of control qubits.
        target (qubit): target qubit.

    """
    if n_controls == 0:
        x(target)

    elif n_controls == 1:
        cx(controls[0], target)

    elif n_controls == 2:
        toffoli(controls[0], controls[1], target)

    elif n_controls < 6:
        allocated_ancilla = qubit()

        _log_ladder_one_ancilla(
            controls,
            target,
            allocated_ancilla,
            False,
        )

        discard(allocated_ancilla)

    else:
        allocated_ancilla = qubit()
        action_ancilla = qubit()

        _log_ladder_two_ancillas(
            controls,
            target,
            allocated_ancilla,
            action_ancilla,
            False,
        )

        discard(action_ancilla)
        discard(allocated_ancilla)


@guppy
@no_type_check
def cnx_cca_logdepth_dirty[n_controls: nat](
    controls: array[qubit, n_controls],
    target: qubit,
    borrowed_a: qubit,
    borrowed_b: qubit,
) -> None:
    r"""Apply a logarithmic-depth CCA-based cnx using 2 dirty ancillae.

    Based on Section 5.5 of https://arxiv.org/abs/2407.17966

    Args:
        controls (array[qubit, n_controls]): register of control qubits.
        target (qubit): target qubit.
        borrowed_a (qubit): first borrowed ancilla.
        borrowed_b (qubit): second borrowed ancilla.

    """
    if n_controls == 0:
        x(target)

    elif n_controls == 1:
        cx(controls[0], target)

    elif n_controls == 2:
        toffoli(controls[0], controls[1], target)

    elif n_controls < 6:
        _log_ladder_one_ancilla(controls, target, borrowed_a, True)
        _log_ladder_one_ancilla(controls, target, borrowed_a, True)

    else:
        _log_ladder_two_ancillas(controls, target, borrowed_a, borrowed_b, True)
        _log_ladder_two_ancillas(controls, target, borrowed_a, borrowed_b, True)
