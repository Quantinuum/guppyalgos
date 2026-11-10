"""Increment a qubit register."""

from typing import no_type_check

from guppylang.decorator import guppy
from guppylang.std.builtins import array, comptime, nat, dagger
from guppylang.std.quantum import cx, discard, qubit, x

from guppyalgos.primitives.gate_decompositions.and_op import temp_and


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


@guppy
@no_type_check
def _get_max_value(n: nat @ comptime) -> int:
    result = 1
    for _ in range(n):
        result = result * 2

    return result


@guppy
@no_type_check
def linear_depth_incrementer_1[n_q: nat](
    q: array[qubit, n_q],
) -> None:
    """Call the incrementer using the full register range."""
    linear_depth_incrementer(q, _get_max_value(n_q))


@guppy
@no_type_check
def linear_depth_incrementer_2[n_q: nat](
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


@guppy.overload(linear_depth_incrementer_1, linear_depth_incrementer_2)
def linear_depth_incrementer() -> None:
    """Support incrementer calls with or without `max_value`."""
    ...


@guppy
@no_type_check
def cntrl_linear_depth_incrementer_1[n_q: nat](
    control: qubit,
    q: array[qubit, n_q],
) -> None:
    """Call the controlled incrementer using the full register range."""
    cntrl_linear_depth_incrementer(control, q, _get_max_value(n_q))


@guppy
@no_type_check
def cntrl_linear_depth_incrementer_2[n_q: nat](
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


@guppy.overload(cntrl_linear_depth_incrementer_1, cntrl_linear_depth_incrementer_2)
def cntrl_linear_depth_incrementer() -> None:
    """Overloaded function to support different signatures."""
    ...


@guppy
@no_type_check
def linear_depth_decrementer_1[n_q: nat](
    q: array[qubit, n_q],
) -> None:
    """Call the decrementer using the full register range."""
    linear_depth_decrementer(q, _get_max_value(n_q))


@guppy
@no_type_check
def linear_depth_decrementer_2[n_q: nat](
    q: array[qubit, n_q],
    max_value: int,
) -> None:
    """Decrement a little-endian register via X-increment-X.

    Args:
        q (array[qubit, n_q]): qubit array to decrement
        max_value (int): maximum value held on q at the time this function is called.

    """
    active_bits = _compute_active_bits(max_value, n_q)

    for i in range(active_bits):
        x(q[i])
    linear_depth_incrementer(q, max_value)
    for i in range(active_bits):
        x(q[i])


@guppy.overload(linear_depth_decrementer_1, linear_depth_decrementer_2)
def linear_depth_decrementer() -> None:
    """Support decrementer calls with or without `max_value`."""
    ...


@guppy
@no_type_check
def cntrl_linear_depth_decrementer_1[n_q: nat](
    control: qubit,
    q: array[qubit, n_q],
) -> None:
    """Call the controlled decrementer using the full register range."""
    cntrl_linear_depth_decrementer(control, q, _get_max_value(n_q))


@guppy
@no_type_check
def cntrl_linear_depth_decrementer_2[n_q: nat](
    control: qubit,
    q: array[qubit, n_q],
    max_value: int,
) -> None:
    """Decrement a quantum register iff a control qubit is 1.

    Args:
        control (qubit): control qubit
        q (array[qubit, n_q]): qubit array to decrement
        max_value (int): maximum value held on q at the time this function is called.

    """
    active_bits = _compute_active_bits(max_value, n_q)

    for i in range(active_bits):
        x(q[i])
    cntrl_linear_depth_incrementer(control, q, max_value)
    for i in range(active_bits):
        x(q[i])


@guppy.overload(cntrl_linear_depth_decrementer_1, cntrl_linear_depth_decrementer_2)
def cntrl_linear_depth_decrementer() -> None:
    """Support controlled decrementer calls with or without `max_value`."""
    ...
