"""Temporary indexed AND operation with a custom dagger."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import Unitary, array, control, dagger, nat
from guppylang.std.quantum import qubit

from guppyalgos.primitives.gate_decompositions.and_op.utils import index_and


@guppy.unitary
class temp_and_index:  # noqa: D101 -- Guppy requires the docstring on __call__.
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
        r"""Compute an indexed AND into a clean, borrowed target qubit.

        Args:
            target_q: Clean target qubit storing the result.
            control_q0: First control qubit.
            control_0_ind: Required value of the first control.
            control_q1: Second control qubit.
            control_1_ind: Required value of the second control.
            and_operation: AND unitary with its corresponding uncompute modifier.

        """
        index_and(
            target_q,
            control_q0,
            control_0_ind,
            control_q1,
            control_1_ind,
            and_operation,
        )

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
        """Uncompute the indexed AND and return the borrowed target to zero."""
        with dagger:
            index_and(
                target_q,
                control_q0,
                control_0_ind,
                control_q1,
                control_1_ind,
                and_operation,
            )

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
        """Compute the indexed AND conditioned on additional controls."""
        with control(controls):
            index_and(
                target_q,
                control_q0,
                control_0_ind,
                control_q1,
                control_1_ind,
                and_operation,
            )

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
        """Uncompute the controlled indexed AND and return its target to zero."""
        with control(controls), dagger:
            index_and(
                target_q,
                control_q0,
                control_0_ind,
                control_q1,
                control_1_ind,
                and_operation,
            )
