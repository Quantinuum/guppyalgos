"""Temporary logical AND with custom control and measurement-based dagger."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, dagger, nat
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

from guppyalgos.utils import qarray


@guppy.unitary
class temp_and:  # noqa: D101 -- Guppy requires the docstring on __call__.
    @guppy
    @no_type_check
    def __call__(q0: qubit, q1: qubit, target_q: qubit) -> None:
        r"""Compute an AND into a clean target, with measurement-based uncomputation.

        The forward operation uses four T gates following
        https://arxiv.org/abs/1709.06648. Its target must start in $\ket{0}$.
        The custom dagger implements Fig. 4 of https://arxiv.org/abs/1805.03662:
        it requires the target to hold the AND and returns it to $\ket{0}$.

        Under ``control``, the target holds the AND of both inputs and all added
        controls. Each added control uses one clean work qubit and four T gates;
        work qubits are uncomputed using the measurement-based dagger. The controlled
        dagger requires the corresponding controlled AND in its target.

        These are promised-input operations: the measurement-based dagger is not
        an inverse on arbitrary target states.
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
        if n == 0:
            temp_and(q0, q1, target_q)
            return

        work = qarray(n)
        temp_and(controls[0], q0, work[0])
        for i in range(1, n):
            temp_and(controls[i], work[i - 1], work[i])
        temp_and(work[n - 1], q1, target_q)
        for i in range(n - 1, 0, -1):
            with dagger:
                temp_and(controls[i], work[i - 1], work[i])
        with dagger:
            temp_and(controls[0], q0, work[0])
        discard_array(work)

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat](
        q0: qubit, q1: qubit, target_q: qubit, controls: array[qubit, n]
    ) -> None:
        """Uncompute the controlled AND and return its target to zero."""
        if n == 0:
            with dagger:
                temp_and(q0, q1, target_q)
            return

        work = qarray(n)
        temp_and(controls[0], q0, work[0])
        for i in range(1, n):
            temp_and(controls[i], work[i - 1], work[i])
        with dagger:
            temp_and(work[n - 1], q1, target_q)
        for i in range(n - 1, 0, -1):
            with dagger:
                temp_and(controls[i], work[i - 1], work[i])
        with dagger:
            temp_and(controls[0], q0, work[0])
        discard_array(work)
