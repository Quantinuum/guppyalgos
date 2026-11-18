"""Indexed temporary AND."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import Unitary, array, control, dagger, nat
from guppylang.std.quantum import qubit, x


@guppy.unitary
class temp_and_indexed:  # noqa: D101 -- Guppy requires the docstring on __call__.
    @guppy
    @no_type_check
    def __call__(
        target_q: qubit,
        control_q0: qubit,
        control_0_ind: bool,
        control_q1: qubit,
        control_1_ind: bool,
        and_operation: Unitary[[qubit, qubit, qubit], None],
    ) -> None:
        """Apply an AND operation with controls matching the given indicators.

        Args:
            target_q: Target qubit storing the result.
            control_q0: First control qubit.
            control_0_ind: Required value of the first control.
            control_q1: Second control qubit.
            control_1_ind: Required value of the second control.
            and_operation: AND unitary, including its dagger and control modifiers.

        """
        if not control_0_ind:
            x(control_q0)
        if not control_1_ind:
            x(control_q1)
        and_operation(control_q0, control_q1, target_q)
        if not control_0_ind:
            x(control_q0)
        if not control_1_ind:
            x(control_q1)

    @guppy
    @no_type_check
    def daggered(
        target_q: qubit,
        control_q0: qubit,
        control_0_ind: bool,
        control_q1: qubit,
        control_1_ind: bool,
        and_operation: Unitary[[qubit, qubit, qubit], None],
    ) -> None:
        """Apply the AND's dagger with the same indexed controls."""
        if not control_0_ind:
            x(control_q0)
        if not control_1_ind:
            x(control_q1)
        with dagger:
            and_operation(control_q0, control_q1, target_q)
        if not control_0_ind:
            x(control_q0)
        if not control_1_ind:
            x(control_q1)

    @guppy
    @no_type_check
    def controlled[n: nat](
        target_q: qubit,
        control_q0: qubit,
        control_0_ind: bool,
        control_q1: qubit,
        control_1_ind: bool,
        and_operation: Unitary[[qubit, qubit, qubit], None],
        controls: array[qubit, n],
    ) -> None:
        """Apply the indexed AND conditioned on all additional controls."""
        if not control_0_ind:
            x(control_q0)
        if not control_1_ind:
            x(control_q1)
        with control(controls):
            and_operation(control_q0, control_q1, target_q)
        if not control_0_ind:
            x(control_q0)
        if not control_1_ind:
            x(control_q1)

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat](
        target_q: qubit,
        control_q0: qubit,
        control_0_ind: bool,
        control_q1: qubit,
        control_1_ind: bool,
        and_operation: Unitary[[qubit, qubit, qubit], None],
        controls: array[qubit, n],
    ) -> None:
        """Apply the indexed AND's dagger with additional controls."""
        if not control_0_ind:
            x(control_q0)
        if not control_1_ind:
            x(control_q1)
        with control(controls), dagger:
            and_operation(control_q0, control_q1, target_q)
        if not control_0_ind:
            x(control_q0)
        if not control_1_ind:
            x(control_q1)


temp_and_index = temp_and_indexed
