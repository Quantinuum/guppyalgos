"""Utility for allocating qubit arrays."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.quantum import qubit


@guppy
@no_type_check
def qarray(n: nat @ comptime) -> "array[qubit, n]":  # noqa: F821
    """Allocate a qubit array of length n."""
    return array(qubit() for _ in range(n))
