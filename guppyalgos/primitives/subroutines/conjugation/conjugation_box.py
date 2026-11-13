"""Conjugation of a unitary by a daggerable function."""

from __future__ import annotations

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import Daggerable, Unitary, array, control, dagger, nat
from guppylang.std.quantum import qubit


@guppy.struct
@no_type_check
class ConjugationBox[
    Regs,
    G: Daggerable[[Regs], None],  # ty: ignore[invalid-type-variable-bound]
    U: Unitary[[Regs], None],  # ty: ignore[invalid-type-variable-bound]
]:
    """Store G and U and apply G, U, then G† to a shared register."""

    g: G
    u: U

    @guppy.unitary
    class compose:  # noqa: D106 - Guppy unitary classes permit only methods.
        @guppy
        @no_type_check
        def __call__(self, qs: Regs) -> None:
            """Apply G, U, then G†."""
            self.g(qs)
            self.u(qs)
            with dagger:
                self.g(qs)

        @guppy
        @no_type_check
        def daggered(self, qs: Regs) -> None:
            """Apply G, U†, then G†."""
            self.g(qs)
            with dagger:
                self.u(qs)
            with dagger:
                self.g(qs)

        @guppy
        @no_type_check
        def controlled[n: nat](self, qs: Regs, controls: array[qubit, n]) -> None:
            """Apply G, controlled-U, then G†."""
            self.g(qs)
            with control(controls):
                self.u(qs)
            with dagger:
                self.g(qs)

        @guppy
        @no_type_check
        def ctrl_daggered[n: nat](self, qs: Regs, controls: array[qubit, n]) -> None:
            """Apply G, controlled-U†, then G†."""
            self.g(qs)
            with control(controls), dagger:
                self.u(qs)
            with dagger:
                self.g(qs)
