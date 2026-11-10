"""Utility functions for AND operations."""

from guppylang import guppy

from guppylang.std.quantum import (
    qubit,
    x,
)
from guppylang.std.builtins import Unitary, dagger


from typing import no_type_check


@guppy.unitary
class index_and:  # noqa: D101 -- Guppy requires the docstring on __call__.
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
        """Perform an AND operation on indexed control qubits and store the result.

        Args:
            target_q (qubit): The target qubit to store the result.
            control_q0 (qubit): The first control qubit.
            control_0_ind (bool): Indicator for the first control qubit.
            control_q1 (qubit): The second control qubit.
            control_1_ind (bool): Indicator for the second control qubit.
            and_operation (Unitary[[qubit, qubit, qubit], None]): The AND operation
                to apply.

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
        """Uncompute an indexed AND using the supplied unitary's dagger."""
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
