"""General utils for guppyalgos."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import (
    Function,
    Unitary,
    array,
    comptime,
    control,
    dagger,
    frozenarray,
    nat,
)
from guppylang.std.quantum import (
    crz,
    cx,
    discard,
    h,
    qubit,
    rz,
    t,
    toffoli,
    x,
)


from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx


@guppy.unitary
class transversal1:  # noqa: D101
    @guppy
    @no_type_check
    def __call__[n: nat, m: nat](
        op: Unitary[[qubit], None], qs: array[qubit, n], ignore_: array[nat, m]
    ) -> None:
        """Apply single qubit unitary register-wise ignoring the indices in ``ignore_``.

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
    def daggered[n: nat, m: nat](  # noqa: D102
        op: Unitary[[qubit], None], qs: array[qubit, n], ignore_: array[nat, m]
    ) -> None:
        for i in range(n):
            apply_gate = True
            for j in range(m):
                if i == ignore_[j]:
                    apply_gate = False
            if apply_gate:
                with dagger:
                    op(qs[i])

    @guppy
    @no_type_check
    def controlled[n: nat, m: nat, n_ctrl: nat](  # noqa: D102
        op: Unitary[[qubit], None],
        qs: array[qubit, n],
        ignore_: array[nat, m],
        ctrls: array[qubit, n_ctrl],
    ) -> None:
        for i in range(n):
            apply_gate = True
            for j in range(m):
                if i == ignore_[j]:
                    apply_gate = False
            if apply_gate:
                with control(ctrls):
                    op(qs[i])

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat, m: nat, n_ctrl: nat](  # noqa: D102
        op: Unitary[[qubit], None],
        qs: array[qubit, n],
        ignore_: array[nat, m],
        ctrls: array[qubit, n_ctrl],
    ) -> None:
        for i in range(n):
            apply_gate = True
            for j in range(m):
                if i == ignore_[j]:
                    apply_gate = False
            if apply_gate:
                with control(ctrls), dagger:
                    op(qs[i])


@guppy.unitary
class transversal1_start_end:  # noqa: D101
    @guppy
    @no_type_check
    def __call__[n: nat](
        op: Unitary[[qubit], None],
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
    def daggered[n: nat](  # noqa: D102
        op: Unitary[[qubit], None],
        qs: array[qubit, n],
        start: nat,
        end: nat,
    ) -> None:
        for i in range(start, end):
            with dagger:
                op(qs[i])

    @guppy
    @no_type_check
    def controlled[n: nat, n_ctrl: nat](  # noqa: D102
        op: Unitary[[qubit], None],
        qs: array[qubit, n],
        start: nat,
        end: nat,
        ctrls: array[qubit, n_ctrl],
    ) -> None:
        for i in range(start, end):
            with control(ctrls):
                op(qs[i])

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat, n_ctrl: nat](  # noqa: D102
        op: Unitary[[qubit], None],
        qs: array[qubit, n],
        start: nat,
        end: nat,
        ctrls: array[qubit, n_ctrl],
    ) -> None:
        for i in range(start, end):
            with control(ctrls), dagger:
                op(qs[i])


@guppy(unitary=True)
@no_type_check
def transversal1_all[n: nat](op: Unitary[[qubit], None], qs: array[qubit, n]) -> None:
    """Apply a single qubit operator register-wise.

    Args:
        op: The operator to apply.
        qs: The qubit array to apply the operator to.

    """
    transversal1(op, qs, array())


@guppy.unitary
class transversal2:  # noqa: D101
    @guppy
    @no_type_check
    def __call__[n: nat, m: nat](
        op: Unitary[[qubit, qubit], None],
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
    def daggered[n: nat, m: nat](  # noqa: D102
        op: Unitary[[qubit, qubit], None],
        qs0: array[qubit, n],
        qs1: array[qubit, n],
        ignore_: array[nat, m],
    ) -> None:
        for i in range(n):
            apply_gate = True
            for j in range(m):
                if i == ignore_[j]:
                    apply_gate = False
            if apply_gate:
                with dagger:
                    op(qs0[i], qs1[i])

    @guppy
    @no_type_check
    def controlled[n: nat, m: nat, n_ctrl: nat](  # noqa: D102
        op: Unitary[[qubit, qubit], None],
        qs0: array[qubit, n],
        qs1: array[qubit, n],
        ignore_: array[nat, m],
        ctrls: array[qubit, n_ctrl],
    ) -> None:
        for i in range(n):
            apply_gate = True
            for j in range(m):
                if i == ignore_[j]:
                    apply_gate = False
            if apply_gate:
                with control(ctrls):
                    op(qs0[i], qs1[i])

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat, m: nat, n_ctrl: nat](  # noqa: D102
        op: Unitary[[qubit, qubit], None],
        qs0: array[qubit, n],
        qs1: array[qubit, n],
        ignore_: array[nat, m],
        ctrls: array[qubit, n_ctrl],
    ) -> None:
        for i in range(n):
            apply_gate = True
            for j in range(m):
                if i == ignore_[j]:
                    apply_gate = False
            if apply_gate:
                with control(ctrls), dagger:
                    op(qs0[i], qs1[i])


@guppy.unitary
class traversal2_start_end:  # noqa: D101
    @guppy
    @no_type_check
    def __call__[n: nat](
        op: Unitary[[qubit, qubit], None],
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
    def daggered[n: nat](  # noqa: D102
        op: Unitary[[qubit, qubit], None],
        qs0: array[qubit, n],
        qs1: array[qubit, n],
        start: nat,
        end: nat,
    ) -> None:
        for i in range(start, end):
            with dagger:
                op(qs0[i], qs1[i])

    @guppy
    @no_type_check
    def controlled[n: nat, n_ctrl: nat](  # noqa: D102
        op: Unitary[[qubit, qubit], None],
        qs0: array[qubit, n],
        qs1: array[qubit, n],
        start: nat,
        end: nat,
        ctrls: array[qubit, n_ctrl],
    ) -> None:
        for i in range(start, end):
            with control(ctrls):
                op(qs0[i], qs1[i])

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat, n_ctrl: nat](  # noqa: D102
        op: Unitary[[qubit, qubit], None],
        qs0: array[qubit, n],
        qs1: array[qubit, n],
        start: nat,
        end: nat,
        ctrls: array[qubit, n_ctrl],
    ) -> None:
        for i in range(start, end):
            with control(ctrls), dagger:
                op(qs0[i], qs1[i])


@guppy(unitary=True)
@no_type_check
def transversal2_all[n: nat](
    op: Unitary[[qubit, qubit], None],
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


@guppy
@no_type_check
def apply_phase(q: qubit, theta: angle) -> None:
    """Apply a phase rotation to a qubit."""
    rz(q, theta)


@guppy(unitary=True)
@no_type_check
def t_state(qubit: qubit) -> None:
    """Directly generate a T state on a qubit."""
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


@guppy
@no_type_check
def _swap_qubit_controlled[n_ctrl: nat](
    a: qubit,
    b: qubit,
    ctrls: array[qubit, n_ctrl],
    cnx_kernel: Function[[array[qubit, n_ctrl], qubit], None],
) -> None:
    cx(a, b)
    if n_ctrl == 1:
        toffoli(ctrls[0], b, a)
    else:
        work = qubit()
        cnx_kernel(ctrls, work)
        toffoli(work, b, a)
        cnx_kernel(ctrls, work)
        discard(work)
    cx(a, b)


@guppy.unitary
class swap_qubit:  # noqa: D101
    @guppy
    @no_type_check
    def __call__(a: qubit, b: qubit) -> None:
        """Swap two qubits.

        The unmodified version just exchanges labels of the qubits at the computational
        level, but when controlled this performs a cswap gate.

        Note that even the non-modified version may have a non-trivial cost depending
        on the QEC code compiled to.
        """
        mem_swap(a, b)

    @guppy
    @no_type_check
    def daggered(a: qubit, b: qubit) -> None:  # noqa: D102
        swap_qubit(a, b)

    @guppy
    @no_type_check
    def controlled[n_ctrl: nat](  # noqa: D102
        a: qubit, b: qubit, ctrls: array[qubit, n_ctrl]
    ) -> None:
        _swap_qubit_controlled(a, b, ctrls, cnx[n_ctrl])

    @guppy
    @no_type_check
    def ctrl_daggered[n_ctrl: nat](  # noqa: D102
        a: qubit, b: qubit, ctrls: array[qubit, n_ctrl]
    ) -> None:
        _swap_qubit_controlled(a, b, ctrls, cnx[n_ctrl])


@guppy(unitary=True)
@no_type_check
def swap_reg[n: nat](a: array[qubit, n], b: array[qubit, n]) -> None:
    """Swap two registers.

    The unmodified version just exchanges labels of the qubits at the computational
    level, but when controlled this performs cswap gates.

    Note that even the non-modified version may have a non-trivial cost depending
    on the QEC code compiled to.
    """
    transversal(swap_qubit, a, b)


@guppy.overload(swap_qubit, swap_reg)
def swap() -> None:
    """Swap qubits or registers.

    The unmodified version just exchanges labels of the qubits at the computational
    level, but when controlled this performs cswap gates.

    Note that even the non-modified version may have a non-trivial cost depending
    on the QEC code compiled to.
    """
    ...


# Explicit gates for cswaps to avoid modifier lowering bugs
# and struct-field borrowing bugs: https://github.com/Quantinuum/guppylang/issues/2466.
# Plain calls also avoid unsupported modifiers inside comptime functions:
# https://github.com/Quantinuum/guppylang/issues/2461.
@guppy(daggerable=True)
@no_type_check
def _cswap_qubit(c: qubit, a: qubit, b: qubit) -> None:
    """Apply a controlled swap without modifiers."""
    cx(a, b)
    toffoli(c, b, a)
    cx(a, b)


@guppy.comptime(daggerable=True)
@no_type_check
def _cswap_register_impl[n: nat](
    c: qubit, a: array[qubit, n], b: array[qubit, n]
) -> None:
    """Apply controlled swaps register-wise without modifiers."""
    for i in range(n):
        _cswap_qubit(c, a[i], b[i])


@guppy(daggerable=True)
@no_type_check
def _cswap_register[n: nat](c: qubit, a: array[qubit, n], b: array[qubit, n]) -> None:
    """Apply daggerable controlled register swap."""
    _cswap_register_impl(c, a, b)


@guppy.overload(_cswap_qubit, _cswap_register)
def _cswap() -> None:
    """Apply controlled-swap workaround to qubits or registers."""
    ...


@guppy
@no_type_check
def _ccswap[n: nat](
    c0: qubit, c1: qubit, a: array[qubit, n], b: array[qubit, n]
) -> None:
    """Swap registers conditioned on both controls using a work qubit."""
    work = qubit()
    toffoli(c0, c1, work)
    _cswap(work, a, b)
    toffoli(c0, c1, work)
    discard(work)


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
