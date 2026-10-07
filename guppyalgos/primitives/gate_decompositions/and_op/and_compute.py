"""compute AND operation implementations."""

from guppylang import guppy
from guppylang.std.quantum import qubit, t, tdg, cx, h, s

from guppyalgos.utils import t_state


from typing import no_type_check


@guppy
@no_type_check
def temp_and_t_state_compute(q0: qubit, q1: qubit, t_qubit: qubit) -> None:
    r"""Temporary AND computation acting on T state target.

    Following the T-depth 1 construction in https://arxiv.org/abs/1709.06648
    which uses 3 T-gates and 1 incoming $|T>$ state.

    Args:
        q0 (qubit): The first input qubit.
        q1 (qubit): The second input qubit.
        t_qubit (qubit): The target qubit to store the result.
            Must be in the $|T>$ state.

    """
    # incoming |T> state on t_qubit
    cx(q0, t_qubit)
    cx(q1, t_qubit)

    cx(t_qubit, q0)
    cx(t_qubit, q1)

    tdg(q0)
    tdg(q1)
    t(t_qubit)

    cx(t_qubit, q0)
    cx(t_qubit, q1)

    h(t_qubit)
    s(t_qubit)


@guppy
@no_type_check
def temp_and_compute(q0: qubit, q1: qubit, t_qubit: qubit) -> None:
    r"""Temporary AND computation acting on 0 state target.

    Following the construction in https://arxiv.org/abs/1805.03662
    which uses 4 T-gates.

    Args:
        q0 (qubit): The first input qubit.
        q1 (qubit): The second input qubit.
        t_qubit (qubit): The target qubit to store the result. Begins in $\ket{0}$

    """
    t_state(t_qubit)
    temp_and_t_state_compute(q0, q1, t_qubit)
