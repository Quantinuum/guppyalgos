"""Tests for unsafe borrow utils."""

from typing import no_type_check

import pytest
from guppylang import guppy
from guppylang.emulator import EmulatorError
from guppylang.std.builtins import output
from guppylang.std.quantum import (
    collect_measurements,
    discard,
    discard_array,
    measure_array,
    x,
)

from guppyalgos.utils import qarray
from guppyalgos.utils import _unsafe_array_borrow, _unsafe_array_unborrow


def test_unsafe_array_borrow_order() -> None:
    """Test unsafe array borrow returns bits in correct order."""

    @guppy
    def main() -> None:
        qs = qarray(2)  # not owned input
        x(qs[0])
        ps = _unsafe_array_borrow(qs)
        # ps now contains the qubits of qs, but is owned
        _unsafe_array_unborrow(qs, ps)
        output("meas", collect_measurements(measure_array(qs)))

    res = main.emulator(2).run()
    assert res.results[0][0] == ("meas", [1, 0])


def test_unsafe_array_borrow_borrowed_again_error() -> None:
    """Test unsafe array borrow error on second borrow."""

    @guppy
    @no_type_check
    def main() -> None:
        qs = qarray(2)
        ps = _unsafe_array_borrow(qs)
        discard(ps.take(0))
        _unsafe_array_unborrow(qs, ps)
        discard_array(qs)

    with pytest.raises(EmulatorError, match="borrowed again"):
        main.emulator(2).run()


def test_unsafe_array_borrow_non_empty_return() -> None:
    """Test unsafe array borrow error on empty_target being non-empty."""

    @guppy
    @no_type_check
    def main() -> None:
        qs = qarray(2)
        ps = _unsafe_array_borrow(qs)
        qs.put((ps.take(0)), 0)
        _unsafe_array_unborrow(qs, ps)
        discard_array(qs)

    with pytest.raises(EmulatorError, match="still contains qubits"):
        main.emulator(2).run()
