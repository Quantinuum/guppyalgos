"""Guppy runtime math functions."""

from typing import no_type_check

from guppylang import comptime, guppy
from guppylang.std.builtins import array, nat


@guppy
@no_type_check
def get_bit(k: int, i: int) -> bool:
    """Extract bit i from integer k.

    Args:
        k: Integer value.
        i: Bit position, where 0 is the rightmost (least significant) bit.

    Returns:
        bool: True if bit i is 1, False otherwise.

    """
    return ((k >> i) & 1) == 1


@guppy
@no_type_check
def get_bits(k: int, length: nat @ comptime) -> "array[bool, length]":  # noqa: F821
    """Get the first n (little endian) bits of integer k.

    Args:
        k: Integer value.
        length: length of the bitstring to be returned

    Returns:
        array[bool, length]: little endian bits representing the integer k

    """
    return array(get_bit(k, i) for i in range(length))


@guppy
@no_type_check
def floor(x: float) -> int:
    """Compute the largest integer less than or equal to x.

    Args:
        x: Floating-point value.

    Returns:
        int: Largest integer less than or equal to x.

    """
    i = int(x)
    if float(i) > x:
        return i - 1
    return i
