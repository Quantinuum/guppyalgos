"""Tests for classical util methods."""

import pytest

from guppyalgos.utils.python.binary import (
    bits_to_int,
    fixed_point_to_float,
    float_to_fixed_point,
    int_to_bits,
)


def test_int_to_bits() -> None:
    """Test int to little endian bitstring conversion."""
    assert int_to_bits(0, 3) == [False, False, False]
    assert int_to_bits(1, 3) == [True, False, False]
    assert int_to_bits(3, 3) == [True, True, False]
    assert int_to_bits(5, 10) == [True, False, True] + 7 * [False]
    assert int_to_bits(-8, 4, signed=True) == [False, False, False, True]
    assert int_to_bits(-2, 6, signed=True) == [False] + 5 * [True]


def test_int_to_bits_invalid() -> None:
    """Test int to bit raises when bits cannot be represented."""
    with pytest.raises(ValueError, match="not able to represent"):
        int_to_bits(-9, 4, signed=True)
    with pytest.raises(ValueError, match="not able to represent"):
        int_to_bits(8, 3)


def test_float_to_fixed_point() -> None:
    """Test floating to fixed point conversion."""
    assert float_to_fixed_point(0.5, 1) == [True]
    assert float_to_fixed_point(0.75, 2) == [True, True]
    assert float_to_fixed_point(0.25, 2) == [True, False]
    assert float_to_fixed_point(0.25, 4) == [False, False, True, False]
    assert float_to_fixed_point(0.74, 2) == [True, True]
    assert float_to_fixed_point(0.99, 2) == [True, True]
    assert float_to_fixed_point(1.375, 3, int_bits=1) == [True, True, False, True]
    assert float_to_fixed_point(2.75, 2, int_bits=2) == [True, True, False, True]
    assert float_to_fixed_point(10.625, 3, int_bits=4) == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
    ]


def test_fixed_point_to_float() -> None:
    """Test fixed point decoding with integer and fractional bits."""
    assert fixed_point_to_float([True], int_bits=0) == 0.5
    assert fixed_point_to_float([True, True], int_bits=0) == 0.75
    assert fixed_point_to_float([False, True, False, False], int_bits=0) == 0.125
    assert fixed_point_to_float([True, False, True, True], int_bits=1) == 1.625
    assert fixed_point_to_float([True, False, True, True], int_bits=2) == 3.25
    assert (
        fixed_point_to_float([True, False, True, False, True, False, True], int_bits=4)
        == 10.625
    )


def test_bits_to_int() -> None:
    """Test little-endian bit-to-integer conversion."""
    assert bits_to_int([False, False, False, False]) == 0
    assert bits_to_int([False, True, True, False]) == 6
    assert bits_to_int([True, False, True, True]) == 13
    assert bits_to_int([False, False, False, True], signed=True) == -8
    assert bits_to_int([False, True, True, True, True, True], signed=True) == -2


def test_little_endian_round_trip() -> None:
    """Test little-endian encoding and decoding round trip."""
    for value in range(16):
        bits = int_to_bits(value, 4)
        assert bits_to_int(bits) == value


def test_signed_little_endian_round_trip() -> None:
    """Test signed little-endian encoding and decoding round trip."""
    for value in range(-8, 8):
        bits = int_to_bits(value, 4, signed=True)
        assert bits_to_int(bits, signed=True) == value


@pytest.mark.parametrize(
    ("value", "frac_bits", "int_bits"),
    [
        (0.625, 3, 0),
        (1.375, 3, 1),
        (2.75, 2, 2),
        (10.625, 3, 4),
    ],
)
def test_fixed_point_round_trip(value: float, frac_bits: int, int_bits: int) -> None:
    """Test that fixed-point encoding and decoding round trip."""
    bits = float_to_fixed_point(value, frac_bits, int_bits=int_bits)
    assert fixed_point_to_float(bits, int_bits=int_bits) == value


def test_invalid_float_to_fixed_point() -> None:
    """Test floats outside the representable range raise exceptions."""
    floats = [-1.0, 1.0, 1.2]
    for f in floats:
        with pytest.raises(ValueError, match="Input float"):
            float_to_fixed_point(f, 10)

    with pytest.raises(ValueError, match="Input float"):
        float_to_fixed_point(2.0, 2, int_bits=1)

    with pytest.raises(ValueError, match="Bit counts"):
        float_to_fixed_point(0.5, -1)

    with pytest.raises(ValueError, match="At least one fixed-point bit"):
        float_to_fixed_point(0.0, 0)
