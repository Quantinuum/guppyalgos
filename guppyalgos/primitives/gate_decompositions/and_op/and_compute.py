"""Temporary logical AND with custom control and measurement-based dagger."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, comptime, dagger, nat
from guppylang.std.quantum import (
    cx,
    cz,
    discard_array,
    h,
    project_z,
    qubit,
    reset,
    s,
    t,
    tdg,
)


@guppy.unitary
class temp_and:  # noqa: D101 -- Guppy requires the docstring on __call__.
    @guppy
    @no_type_check
    def __call__(q0: qubit, q1: qubit, target_q: qubit) -> None:
        r"""Compute an AND into a clean target using four T gates.

        Follows https://arxiv.org/abs/1709.06648. The target must start in
        $\ket{0}$. The custom dagger uses the measurement-based cleanup from
        Fig. 4 of https://arxiv.org/abs/1805.03662 and returns the target to zero.

        The dagger requires the target to hold the corresponding AND; it is
        promised cleanup, not an inverse on arbitrary target states.
        """
        h(target_q)
        t(target_q)
        cx(q0, target_q)
        cx(q1, target_q)
        cx(target_q, q0)
        cx(target_q, q1)
        tdg(q0)
        tdg(q1)
        t(target_q)
        cx(target_q, q0)
        cx(target_q, q1)
        h(target_q)
        s(target_q)

    @guppy
    @no_type_check
    def daggered(q_0: qubit, q_1: qubit, target_q: qubit) -> None:
        """Uncompute the AND using measurement and return the target to zero."""
        h(target_q)
        if project_z(target_q).read():
            cz(q_0, q_1)
            reset(target_q)

    @guppy
    @no_type_check
    def controlled[n: nat](
        q0: qubit, q1: qubit, target_q: qubit, controls: array[qubit, n]
    ) -> None:
        """Compute the AND including all additional controls."""
        _controlled_temp_and(q0, q1, target_q, controls, False)

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat](
        q0: qubit, q1: qubit, target_q: qubit, controls: array[qubit, n]
    ) -> None:
        """Uncompute the controlled AND and return its target to zero."""
        _controlled_temp_and(q0, q1, target_q, controls, True)


@guppy
@no_type_check
def _controlled_temp_and[n: nat](
    q0: qubit,
    q1: qubit,
    target_q: qubit,
    controls: array[qubit, n],
    inverse: bool @ comptime,
) -> None:
    """Share the temporary-AND ladder between the controlled modifiers."""
    if n == 0:
        if inverse:
            with dagger:
                temp_and(q0, q1, target_q)
        else:
            temp_and(q0, q1, target_q)
        return

    work = array(qubit() for _ in range(n))
    temp_and(controls[0], q0, work[0])
    for i in range(1, n):
        temp_and(controls[i], work[i - 1], work[i])
    if inverse:
        with dagger:
            temp_and(work[n - 1], q1, target_q)
    else:
        temp_and(work[n - 1], q1, target_q)
    for i in range(n - 1, 0, -1):
        with dagger:
            temp_and(controls[i], work[i - 1], work[i])
    with dagger:
        temp_and(controls[0], q0, work[0])
    discard_array(work)


# Keep existing compute callbacks using the same public unitary.
temp_and_compute = temp_and
