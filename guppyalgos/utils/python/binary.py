"""Utility methods for guppyalgos."""

from collections.abc import Sequence


def int_to_bits(integer: int, length: int, *, signed=False) -> list[bool]:
    """Convert an integer to a little-endian bit string of input length.

    Args:
        integer (int): The integer to convert.
        length (int): The length of the resulting bit string.
        signed (bool): Flag to represent signed vs unsigned ints.

    Returns:
        list[bool]: A list of boolean values representing the bit string.
        When signed gives the twos-complement, where bits[-1] is the sign bit.

    """
    if not signed and not (0 <= integer < 2**length):
        raise ValueError(f"{length} unsigned bits not able to represent {integer}")
    if signed and not (-(2 ** (length - 1)) <= integer < 2 ** (length - 1)):
        raise ValueError(f"{length} signed bits not able to represent {integer}")

    if integer < 0 and signed:
        integer = (1 << length) + integer
    bits = [bool(int(x)) for x in reversed(bin(integer)[2:].zfill(length))]
    return bits


def bits_to_int(bits: list[bool], *, signed: bool = False) -> int:
    """Interpret ``bits`` as a little-endian integer.

    Args:
        bits: Little-endian bit string. When ``signed`` is true, the final bit is
            interpreted as the sign bit of a two's-complement integer.
        signed: Whether to decode a signed two's-complement integer.

    Returns:
        The integer represented by ``bits``.

    """
    value = sum(int(bit) * (2**i) for i, bit in enumerate(bits))
    if signed and bits and bits[-1]:
        value -= 1 << len(bits)
    return value


def float_to_fixed_point(value: float, frac_bits: int, int_bits: int = 0) -> list[bool]:
    """Convert a float to a fixed-point binary representation.

    Args:
        value: Unsigned value to encode.
        frac_bits: Number of bits below the radix point.
        int_bits: Number of bits above the radix point.

    Returns:
        Little-endian fixed-point bit string.

    Examples:
        >>> float_to_fixed_point(0.25, 2)
        [True, False]
        >>> float_to_fixed_point(1.375, 3, int_bits=1)
        [True, True, False, True]

    """
    if frac_bits < 0 or int_bits < 0:
        raise ValueError("Bit counts must be non-negative")
    if frac_bits + int_bits <= 0:
        raise ValueError("At least one fixed-point bit is required")

    max_value = float(1 << int_bits)
    if not (0 <= value < max_value):
        raise ValueError(f"Input float must be in [0, {max_value})")

    scale = 1 << frac_bits
    encoded_int = int(value * scale + 0.5)  # round to nearest
    max_encoded_int = (1 << (int_bits + frac_bits)) - 1
    encoded_int = min(encoded_int, max_encoded_int)
    total_bits = int_bits + frac_bits
    return [bool((encoded_int >> i) & 1) for i in range(total_bits)]


def fixed_point_to_float(bits: Sequence[bool], int_bits: int = 0) -> float:
    """Convert an unsigned fixed-point bit string to a float.

    Args:
        bits: Fixed-point bit string.
        int_bits: Number of bits above the radix point.

    Returns:
        The decoded unsigned fixed-point value.

    Examples:
        >>> fixed_point_to_float([True, False, False], int_bits=1)
        0.25
        >>> fixed_point_to_float([True, True, False, True], int_bits=1)
        1.375

    """
    if int_bits < 0 or int_bits > len(bits):
        raise ValueError("int_bits must be between 0 and len(bits)")

    bits_tuple = tuple(bits)
    value = 0.0
    frac_bits = len(bits_tuple) - int_bits
    for index, bit in enumerate(bits_tuple):
        power = index - frac_bits
        value += int(bit) * (2**power)
    return value
