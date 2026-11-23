"""Elementary Givens rotations."""

from __future__ import annotations

from typing import no_type_check

from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import Unitary, array, control, dagger, nat
from guppylang.std.quantum import cx, h, qubit, rz, sdg


@guppy(daggerable=True)
@no_type_check
def _change_basis(q0: qubit, q1: qubit) -> None:
    """Put both Givens rotation factors into the Z basis."""
    h(q0)
    cx(q0, q1)
    sdg(q0)
    h(q0)
    sdg(q1)
    h(q1)


def givens_with_custom_rz(
    rz_method: Unitary[[qubit, angle], None],
) -> Unitary[[qubit, qubit, angle], None]:
    """Build a two-mode Givens rotation with a supplied RZ implementation.

    The rotation uses Guppy's half-turn angle convention.
    """

    @guppy(unitary=True)
    @no_type_check
    def rotate_z(q0: qubit, q1: qubit, theta: angle) -> None:
        """Apply the two Z rotations in the transformed basis."""
        rz_method(q0, -theta)
        rz_method(q1, -theta)

    @guppy.unitary
    class givens_rotation_fn:
        @guppy
        @no_type_check
        def __call__(q0: qubit, q1: qubit, theta: angle) -> None:
            """Apply the configured two-mode Givens rotation."""
            _change_basis(q0, q1)
            rotate_z(q0, q1, theta)
            with dagger:
                _change_basis(q0, q1)

        @guppy
        @no_type_check
        def daggered(q0: qubit, q1: qubit, theta: angle) -> None:
            """Apply the inverse Givens rotation."""
            _change_basis(q0, q1)
            with dagger:
                rotate_z(q0, q1, theta)
            with dagger:
                _change_basis(q0, q1)

        @guppy
        @no_type_check
        def controlled[n: nat](
            q0: qubit, q1: qubit, theta: angle, ctrls: array[qubit, n]
        ) -> None:
            """Apply external controls only to the Z rotations."""
            _change_basis(q0, q1)
            with control(ctrls):
                rotate_z(q0, q1, theta)
            with dagger:
                _change_basis(q0, q1)

        @guppy
        @no_type_check
        def ctrl_daggered[n: nat](
            q0: qubit, q1: qubit, theta: angle, ctrls: array[qubit, n]
        ) -> None:
            """Apply external controls only to the inverse Z rotations."""
            _change_basis(q0, q1)
            with control(ctrls), dagger:
                rotate_z(q0, q1, theta)
            with dagger:
                _change_basis(q0, q1)

    return givens_rotation_fn


givens_rotation: Unitary[[qubit, qubit, angle], None] = givens_with_custom_rz(rz)
