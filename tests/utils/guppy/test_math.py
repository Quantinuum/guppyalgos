"""Tests for Guppy runtime math functions."""

import pytest
from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import output

from guppyalgos.utils.guppy.math import get_bit, floor


@pytest.mark.parametrize(
    ("k", "i", "expected"),
    [
        (5, 0, True),
        (5, 1, False),
        (0, 0, False),
        (-1, 0, True),
        (-1, 5, True),
        (-2, 0, False),
    ],
)
def test_get_bit(k: int, i: int, expected: bool) -> None:
    """Test get_bit extracts correct bits from integers."""

    @guppy
    @no_type_check
    def test_circuit() -> None:
        output("bit", get_bit(k, i))

    res = test_circuit.emulator(1).run()
    assert res.collated_shots()[0]["bit"][0] == expected


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (3.7, 3),
        (-3.7, -4),
        (4.0, 4),
    ],
)
def test_floor(x: float, expected: int) -> None:
    """Test floor computes correct integer floors."""

    @guppy
    @no_type_check
    def test_circuit() -> None:
        output("floor_val", floor(x))

    res = test_circuit.emulator(1).run()
    assert res.collated_shots()[0]["floor_val"][0] == expected
