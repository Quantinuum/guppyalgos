"""Common protocol for ladder-like gate sequences."""

from enum import Enum

from guppylang.std.builtins import nat
from guppylang import guppy, array, qubit


@guppy.protocol
class Ladder:
    """Protocol for ladders of gates.

    The ladder has an orientation, with gates oriented in a consistent way.
    Here 'ascending' means that the ladder has gates oriented on increasing indices,
    e.g. a linear cx ladder:

    for i in range(n-1):
        cx(qs[i], qs[i+1])

    descending instead has the ladder oriented the other way.
    """

    # TODO these could be staticmethods
    @guppy.require(unitary=True)
    def ascending[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply ladder ascending in qubit index."""
        ...

    @guppy.require(unitary=True)
    def descending[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply ladder descending in qubit index."""
        ...


class LadderIndexing(Enum):
    """Variant of indexing for the 4 ladder impls.

    For use in comptime or python internals.
    """

    ASCENDING = 0
    DESCENDING = 1
    ASCENDING_DAGGER = 2
    DESCENDING_DAGGER = 3
