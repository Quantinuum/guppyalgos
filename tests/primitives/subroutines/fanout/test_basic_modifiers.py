"""Public modifier integration checks for the migrated circuits."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import array, control, dagger, comptime
from guppylang.std.quantum import qubit
from guppyalgos.testing import get_unitary, assert_allclose_ignorephase

from guppyalgos.primitives.subroutines.fanout.fanout_basic import fanout_basic
from guppyalgos.primitives.subroutines.parity.parity import parity_sequential
from guppyalgos.utils.guppy.unsafe_borrow import (
    _unsafe_array_borrow_slice,
    _unsafe_array_unborrow_slice,
)


@pytest.mark.parametrize("mode", ["forward", "dagger", "control", "combined"])
@pytest.mark.parametrize("kind", ["fanout", "parity"])
def test_fanout_parity_modifiers(mode: str, kind: str) -> None:
    """Check phases and every coherent input, restoring all borrowed registers."""

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, 4]) -> None:
        prefix, data, suffix = _unsafe_array_borrow_slice(qs, 2, 2, 0)
        if comptime(mode == "forward"):
            if comptime(kind == "fanout"):
                fanout_basic(prefix[1], data)
            else:
                parity_sequential(prefix[1], data)
        elif comptime(mode == "dagger"):
            if comptime(kind == "fanout"):
                with dagger:
                    fanout_basic(prefix[1], data)
            else:
                with dagger:
                    parity_sequential(prefix[1], data)
        elif comptime(mode == "control"):
            with control(prefix[0]):
                if comptime(kind == "fanout"):
                    fanout_basic(prefix[1], data)
                else:
                    parity_sequential(prefix[1], data)
        else:
            if comptime(kind == "fanout"):
                with control(prefix[0]), dagger:
                    fanout_basic(prefix[1], data)
            else:
                with control(prefix[0]), dagger:
                    parity_sequential(prefix[1], data)
        _unsafe_array_unborrow_slice(qs, prefix, data, suffix)

    expected = np.zeros((16, 16), dtype=complex)
    for column in range(16):
        bits = [(column >> (3 - i)) & 1 for i in range(4)]
        if mode in ("forward", "dagger") or bits[0]:
            if kind == "fanout":
                bits[2] ^= bits[1]
                bits[3] ^= bits[1]
            else:
                bits[1] ^= bits[2] ^ bits[3]
        row = sum(bit << (3 - i) for i, bit in enumerate(bits))
        expected[row, column] = 1
    assert_allclose_ignorephase(get_unitary(circuit, 4), expected)
