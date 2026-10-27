"""Elementary Givens rotations."""

from __future__ import annotations

from typing import no_type_check

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.quantum import cx, h, qubit, rz, s, sdg


def givens_with_custom_rz(
    rz_method: GuppyFunctionDefinition[[qubit, angle], None],
) -> GuppyFunctionDefinition[[qubit, qubit, angle], None]:
    """Build a two-mode Givens rotation with a supplied RZ implementation.

    The returned callable has signature ``(qubit, qubit, angle) -> None`` and
    uses Guppy's half-turn angle convention.
    """

    @guppy
    @no_type_check
    def givens_rotation_fn(q0: qubit, q1: qubit, theta: angle) -> None:
        """Apply the configured two-mode Givens rotation."""
        h(q0)
        cx(q0, q1)

        sdg(q0)
        h(q0)
        rz_method(q0, -theta)
        h(q0)
        s(q0)

        sdg(q1)
        h(q1)
        rz_method(q1, -theta)
        h(q1)
        s(q1)

        cx(q0, q1)
        h(q0)

    return givens_rotation_fn


givens_rotation: GuppyFunctionDefinition[[qubit, qubit, angle], None] = (
    givens_with_custom_rz(rz)
)
