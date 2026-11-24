"""Common protocol for ladder-like gate sequences."""

from enum import Enum

from guppylang import array, guppy, qubit
from guppylang.std.builtins import nat


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

    # TODO all of these could be staticmethods
    @guppy.require
    def ascending[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply ladder ascending in qubit index."""
        ...

    @guppy.require
    def descending[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply ladder descending in qubit index."""
        ...

    @guppy.require
    def ascending_dagger[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply dagger of ascending ladder."""
        ...

    @guppy.require
    def descending_dagger[n: nat](self, qs: array[qubit, n]) -> None:
        """Apply dagger of descending ladder."""
        ...


class LadderIndexing(Enum):
    """Variant of indexing for the 4 ladder impls.

    For use in comptime or python internals.
    """

    ASCENDING = 0
    DESCENDING = 1
    ASCENDING_DAGGER = 2
    DESCENDING_DAGGER = 3
