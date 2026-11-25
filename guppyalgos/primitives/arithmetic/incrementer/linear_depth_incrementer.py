"""Increment a qubit register."""

from typing import no_type_check

from guppylang.decorator import guppy
from guppylang.std.builtins import array, comptime, dagger, nat
from guppylang.std.quantum import cx, discard, qubit, x

from guppyalgos.primitives.gate_decompositions.and_op import temp_and
from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx
from guppyalgos.primitives.gate_decompositions.cnx.cnu import MultiCntrlFunctionParams


@guppy
@no_type_check
def _incrementer_aux[n_q: nat](
    q: array[qubit, n_q],
    carry: qubit,
    bit_idx: int,
    last_idx: int,
) -> None:
    """Propagate the carry to the top of the register and uncompute on return."""
    if bit_idx == last_idx:
        cx(carry, q[bit_idx])
        return

    next_carry = qubit()
    temp_and(carry, q[bit_idx], next_carry)
    _incrementer_aux(q, next_carry, bit_idx + 1, last_idx)
    with dagger:
        temp_and(carry, q[bit_idx], next_carry)
    discard(next_carry)
    cx(carry, q[bit_idx])


@guppy
@no_type_check
def _compute_active_bits(max_value: int, n_q: nat @ comptime) -> int:
    """Compute the number of active bits given the maximum value held on a register."""
    active_bits = 1
    threshold = 1
    while active_bits < n_q and max_value >= threshold:
        active_bits += 1
        threshold = 2 * threshold + 1

    return active_bits


@guppy(unitary=True)
@no_type_check
def linear_depth_incrementer_1[n_q: nat](
    q: array[qubit, n_q],
) -> None:
    """Call the incrementer using the full register range."""
    linear_depth_incrementer(q, comptime(2**n_q - 1))


@guppy
@no_type_check
def _linear_depth_incrementer_kernel[n_q: nat](
    q: array[qubit, n_q],
    max_value: int,
) -> None:
    """Increment a little-endian register using a linear staircase of toffolis.

    Args:
        q (array[qubit, n_q]): qubit array to increment
        max_value (int): maximum value held on q at the time this function is called.
            If $max_value < 2^n_q$ then the circuit can be optimized to use fewer
            toffolis.

    """
    active_bits = _compute_active_bits(max_value, n_q)

    if active_bits == 1:
        x(q[0])
        return

    if active_bits == 2:
        cx(q[0], q[1])
        x(q[0])
        return

    carry = qubit()
    temp_and(q[0], q[1], carry)
    _incrementer_aux(q, carry, 2, active_bits - 1)
    with dagger:
        temp_and(q[0], q[1], carry)
    discard(carry)
    cx(q[0], q[1])
    x(q[0])


@guppy
@no_type_check
def _controlled_incrementer_kernel[n_q: nat](
    control: qubit,
    q: array[qubit, n_q],
    max_value: int,
) -> None:
    """Increment a quantum register iff a control qubit is 1.

    This is equivalent to incrementing the combined little-endian register
    combined with the control and then flipping the control qubit
    back with an X-gate.

    Args:
        control (qubit): control qubit
        q (array[qubit, n_q]): qubit array to increment
        max_value (int): maximum value held on q at the time this function is called.
            If $max_value < 2^n_q$ then the circuit can be optimized to use fewer
            toffolis.

    """
    active_bits = _compute_active_bits(max_value, n_q)

    if active_bits == 1:
        cx(control, q[0])
        return

    carry = qubit()
    temp_and(control, q[0], carry)
    _incrementer_aux(q, carry, 1, active_bits - 1)
    with dagger:
        temp_and(control, q[0], carry)
    discard(carry)
    cx(control, q[0])


@guppy
@no_type_check
def _controlled_incrementer[n_q: nat, n_controls: nat](
    q: array[qubit, n_q],
    max_value: int,
    controls: array[qubit, n_controls],
    inverse: bool @ comptime,
) -> None:
    """Share the optimized carry chain and compute the joint control only once."""
    active_bits = _compute_active_bits(max_value, n_q)
    if inverse:
        for i in range(active_bits):
            x(q[i])

    operation = MultiCntrlFunctionParams[n_controls, array[qubit, n_q], int](
        _controlled_incrementer_kernel[n_q], cnx[n_controls]
    )
    operation.compose(controls, q, max_value)

    if inverse:
        for i in range(active_bits):
            x(q[i])


@guppy.unitary
class linear_depth_incrementer_2[n_q: nat]:  # noqa: D101
    @guppy
    @no_type_check
    def __call__(q: array[qubit, n_q], max_value: int) -> None:
        """Increment the active low bits, preserving higher bits.

        ``max_value`` is the inclusive bound on the input value. The active
        width includes room for ``max_value + 1``, capped at the register width.
        Arithmetic wraps modulo that active width; at least one qubit is required.
        Temporary carries are cleaned with measurement-based temporary ANDs.
        """
        _linear_depth_incrementer_kernel(q, max_value)

    @guppy
    @no_type_check
    def daggered(q: array[qubit, n_q], max_value: int) -> None:
        """Decrement over the same active width using X-increment-X.

        The bound selects the same width in both directions, including underflow.
        Custom reversal preserves measurement-based cleanup in the recursive chain.
        """
        active_bits = _compute_active_bits(max_value, n_q)
        for i in range(active_bits):
            x(q[i])
        linear_depth_incrementer(q, max_value)
        for i in range(active_bits):
            x(q[i])

    @guppy
    @no_type_check
    def controlled[n_controls: nat](
        q: array[qubit, n_q], max_value: int, controls: array[qubit, n_controls]
    ) -> None:
        """Increment iff every external control is set, restoring all workspace."""
        _controlled_incrementer(q, max_value, controls, False)

    @guppy
    @no_type_check
    def ctrl_daggered[n_controls: nat](
        q: array[qubit, n_q], max_value: int, controls: array[qubit, n_controls]
    ) -> None:
        """Decrement iff every external control is set."""
        _controlled_incrementer(q, max_value, controls, True)


@guppy.overload(linear_depth_incrementer_1, linear_depth_incrementer_2)
def linear_depth_incrementer() -> None:
    """Increment with optional `max_value`, supporting `control` and `dagger`."""
    ...
