"""General utils for guppyalgos."""

from collections.abc import Callable

from guppylang import guppy
from guppylang.std.builtins import array, comptime, nat, frozenarray
from guppylang.std.quantum import qubit, t, h, x, cx, toffoli, discard
from guppylang.std.quantum import crz, rz
from guppylang.std.angles import angle

from typing import no_type_check


@guppy
@no_type_check
def transversal1[n: nat, m: nat](
    op: Callable[[qubit], None], qs: array[qubit, n], ignore_: array[nat, m]
) -> None:
    """Apply a single qubit operator register-wise ignoring the indices in ``ignore_``.

    Args:
        op: The operator to apply.
        qs: The qubit array to apply the operator to.
        ignore_: The indices to ignore.

    """
    for i in range(n):
        apply_gate = True
        for j in range(m):
            if i == ignore_[j]:
                apply_gate = False
        if apply_gate:
            op(qs[i])


@guppy
@no_type_check
def transversal1_start_end[n: nat](
    op: Callable[[qubit], None],
    qs: array[qubit, n],
    start: nat,
    end: nat,
) -> None:
    """Apply a single qubit operator to register from start to end (non-inclusive).

    Args:
        op: The operator to apply.
        qs: The qubit array to apply the operator to.
        start: The start index of the slice.
        end: The end index of the slice (non-inclusive).

    """
    for i in range(start, end):
        op(qs[i])


@guppy
@no_type_check
def transversal1_all[n: nat](op: Callable[[qubit], None], qs: array[qubit, n]) -> None:
    """Apply a single qubit operator register-wise.

    Args:
        op: The operator to apply.
        qs: The qubit array to apply the operator to.

    """
    transversal1(op, qs, array())


@guppy
@no_type_check
def transversal2[n: nat, m: nat](
    op: Callable[[qubit, qubit], None],
    qs0: array[qubit, n],
    qs1: array[qubit, n],
    ignore_: array[nat, m],
) -> None:
    """Apply a two qubit operator register-wise ignoring the indices in ``ignore_``.

    Args:
        op: The operator to apply.
        qs0: The first qubit array to apply the operator to.
        qs1: The second qubit array to apply the operator to.
        ignore_: The indices to ignore.

    """
    for i in range(n):
        apply_gate = True
        for j in range(m):
            if i == ignore_[j]:
                apply_gate = False
        if apply_gate:
            op(qs0[i], qs1[i])


@guppy
@no_type_check
def traversal2_start_end[n: nat](
    op: Callable[[qubit, qubit], None],
    qs0: array[qubit, n],
    qs1: array[qubit, n],
    start: nat,
    end: nat,
) -> None:
    """Apply a two qubit operator to register from start to end (non-inclusive).

    Args:
        op: The operator to apply.
        qs0: The first qubit array to apply the operator to.
        qs1: The second qubit array to apply the operator to.
        start: The start index of the slice.
        end: The end index of the slice (non-inclusive).

    """
    for i in range(start, end):
        op(qs0[i], qs1[i])


@guppy
@no_type_check
def transversal2_all[n: nat](
    op: Callable[[qubit, qubit], None],
    qs0: array[qubit, n],
    qs1: array[qubit, n],
) -> None:
    """Apply a two qubit operator register-wise.

    Args:
        op: The operator to apply.
        qs0: The first qubit array to apply the operator to.
        qs1: The second qubit array to apply the operator to.

    """
    transversal2(op, qs0, qs1, array())


@guppy.overload(
    transversal1,
    transversal1_all,
    transversal1_start_end,
    transversal2,
    transversal2_all,
    traversal2_start_end,
)
@no_type_check
def transversal() -> None:
    """Apply a single or two qubit operator register-wise.

    To apply x to all the qubits in the register:
    transversal(x, register)

    To apply cx to all the qubits in the register:
    transversal(cx, reg1, reg2)

    To apply cx to specific qubits in the register:
    transversal(cx, reg1, reg2, array([1, 3]))
    or
    transversal2(cx, controls, targets, array(comptime(n_qubits - 1)))
    """
    ...


@guppy
@no_type_check
def qarray(n: nat @ comptime) -> "array[qubit, n]":
    """Allocate a qubit array of length n."""
    return array(qubit() for _ in range(n))


@guppy(unitary=True)
@no_type_check
def apply_phase(q: qubit, theta: angle) -> None:
    """Apply Rz(theta), including its global phase convention.

    Supports ``with dagger`` and ``with control`` (including their combination).
    """
    rz(q, theta)


@guppy(unitary=True)
@no_type_check
def t_state(qubit: qubit) -> None:
    """Apply H followed by T; prepares a T state from a zero qubit.

    This circuit is unitary on arbitrary inputs and supports dagger and control.
    """
    h(qubit)
    t(qubit)


@guppy
@no_type_check
def _apply_bitstring_frozen[n: nat](
    reg: array[qubit, n], bit_array: frozenarray[bool, n]
) -> None:
    """Apply bit flips to a reg according to bit_array."""
    for i in range(n):
        if bit_array[i]:
            x(reg[i])


@guppy
@no_type_check
def _apply_bitstring_arr[n: nat](
    reg: array[qubit, n], bit_array: array[bool, n]
) -> None:
    """Apply bit flips to a reg according to bit_array."""
    for i in range(n):
        if bit_array[i]:
            x(reg[i])


@guppy.overload(_apply_bitstring_arr, _apply_bitstring_frozen)
@no_type_check
def apply_bitstring() -> None:
    """Apply bit flips to a reg according to bit_array."""
    ...


@guppy(daggerable=True)
@no_type_check
def cswap_qubit(
    c: qubit,
    t1: qubit,
    t2: qubit,
) -> None:
    """Control swap two qubits."""
    cx(t1, t2)
    toffoli(c, t2, t1)
    cx(t1, t2)


@guppy.comptime(daggerable=True)
@no_type_check
def _cswap_register[n: nat](
    c: qubit, t1_reg: array[qubit, n], t2_reg: array[qubit, n]
) -> None:
    for i in range(n):
        cx(t1_reg[i], t2_reg[i])
        toffoli(c, t2_reg[i], t1_reg[i])
        cx(t1_reg[i], t2_reg[i])


@guppy(daggerable=True)
@no_type_check
def cswap_register[n: nat](
    c: qubit, t1_reg: array[qubit, n], t2_reg: array[qubit, n]
) -> None:
    """Control swap two registers."""
    _cswap_register(c, t1_reg, t2_reg)


@guppy.overload(cswap_qubit, cswap_register)
@no_type_check
def cswap() -> None:
    """Apply a control swap between qubits or equal size registers.

    Either:
        cswap(control_bit, qubit_a, qubit_b)
    Or:
        cswap(control_bit, a_reg, b_reg)
    """
    ...


@guppy
@no_type_check
def ccswap_qubit(
    c0: qubit,
    c1: qubit,
    t1: qubit,
    t2: qubit,
) -> None:
    """Double-control swap two qubits using an internal work qubit."""
    work = qubit()
    toffoli(c0, c1, work)
    cswap(work, t1, t2)
    toffoli(c0, c1, work)
    discard(work)


@guppy
@no_type_check
def ccswap_register[n: nat](
    c0: qubit,
    c1: qubit,
    t1_reg: array[qubit, n],
    t2_reg: array[qubit, n],
) -> None:
    """Double-control swap two equal-size registers using an internal work qubit."""
    work = qubit()
    toffoli(c0, c1, work)
    cswap(work, t1_reg, t2_reg)
    toffoli(c0, c1, work)
    discard(work)


@guppy.overload(ccswap_qubit, ccswap_register)
@no_type_check
def ccswap() -> None:
    """Apply a double-control swap between qubits or equal size registers.

    Either:
        ccswap(control_bit_0, control_bit_1, qubit_a, qubit_b)
    Or:
        ccswap(control_bit_0, control_bit_1, a_reg, b_reg)
    """
    ...


@guppy
@no_type_check
def cphase(control: qubit, target: qubit, theta: angle) -> None:
    """Apply a controlled phase rotation (CU1).

    Args:
        control: Control qubit.
        target: Target qubit.
        theta: Angle of rotation.

    Returns:
        None

    """
    rz(control, theta / 2)  # over rotate the control by half
    crz(control, target, theta)
