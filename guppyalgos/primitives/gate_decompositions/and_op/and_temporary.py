"""Temporary AND operation implementations."""

from guppylang import guppy
from guppylang.std.quantum import (
    qubit,
    discard,
)
from guppylang.std.builtins import Unitary, dagger, owned

from guppyalgos.primitives.gate_decompositions.and_op import index_and


from typing import no_type_check


@guppy
@no_type_check
def temp_and_comp_index(
    control_q0: qubit,
    control_0_ind: bool,
    control_q1: qubit,
    control_1_ind: bool,
    and_operation: Unitary[[qubit, qubit, qubit], None],
) -> qubit:
    r"""Compute a temporary AND operation and return the target qubit.

    This function allocates a new qubit to store the result of the AND operation.
    The allocated qubit is in the $\ket{0}$ state. The returned qubit can be later
    uncomputed using the `temp_and_uncomp_index` function and discarded as a work qubit.

    The supplied AND unitary must accept two input qubits and a clean target.

    Args:
        control_q0 (qubit): The first control qubit.
        control_0_ind (bool): Indicator for the first control qubit.
        control_q1 (qubit): The second control qubit.
        control_1_ind (bool): Indicator for the second control qubit.
        and_operation: The AND unitary to apply.

    Returns:
        qubit: The target qubit storing the result.

    """
    target_q: qubit = qubit()
    index_and(
        target_q, control_q0, control_0_ind, control_q1, control_1_ind, and_operation
    )
    return target_q


@guppy
@no_type_check
def temp_and_uncomp_index(
    target_q: qubit @ owned,
    control_q0: qubit,
    control_0_ind: bool,
    control_q1: qubit,
    control_1_ind: bool,
    and_operation: Unitary[[qubit, qubit, qubit], None],
) -> None:
    """Uncompute a temporary AND operation and discard the target qubit.

    The target qubit must be the one returned by the `temp_and_comp_index` function.
    This function uncomputes the AND operation and then discards the target qubit
    as a work qubit.

    The supplied AND unitary must be the one used to compute the target;
    its custom dagger performs the uncomputation.

    Args:
        target_q (qubit @ owned): The target qubit storing the result.
        control_q0 (qubit): The first control qubit.
        control_0_ind (bool): Indicator for the first control qubit.
        control_q1 (qubit): The second control qubit.
        control_1_ind (bool): Indicator for the second control qubit.
        and_operation: The AND unitary whose dagger uncomputes the target.

    """
    with dagger:
        index_and(
            target_q,
            control_q0,
            control_0_ind,
            control_q1,
            control_1_ind,
            and_operation,
        )
    discard(target_q)
