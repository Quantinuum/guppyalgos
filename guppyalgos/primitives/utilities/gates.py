"""General utils for guppyalgos."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import (
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
    qubit,
    rz,
    toffoli,
    x,
)

from guppyalgos.primitives.gate_decompositions.cnx.cnx import (
    CNXPartialAndLadder,
)


@guppy
@no_type_check
def _transversal1_internal_control[n: nat, m: nat](
    op: Unitary[[qubit], None],
    qs: array[qubit, n],
    ignore_: array[nat, m],
    dag: bool @ comptime,
    ctrl: qubit,
) -> None:
    for i in range(n):
        apply_gate = True
        for j in range(m):
            if i == ignore_[j]:
                apply_gate = False
        if apply_gate:
            with control(ctrl):
                if dag:
                    with dagger:
                        op(qs[i])
                else:
                    op(qs[i])


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
        if n_ctrl == 1:
            _transversal1_internal_control(op, qs, ignore_, False, ctrls[0])
        else:
            temp_control = qubit()
            cnx_ancilla = CNXPartialAndLadder.compute(ctrls, temp_control)
            _transversal1_internal_control(op, qs, ignore_, False, temp_control)
            CNXPartialAndLadder.uncompute(ctrls, cnx_ancilla, temp_control)
            discard(temp_control)

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat, m: nat, n_ctrl: nat](  # noqa: D102
        op: Unitary[[qubit], None],
        qs: array[qubit, n],
        ignore_: array[nat, m],
        ctrls: array[qubit, n_ctrl],
    ) -> None:
        if n_ctrl == 1:
            _transversal1_internal_control(op, qs, ignore_, True, ctrls[0])
        else:
            temp_control = qubit()
            cnx_ancilla = CNXPartialAndLadder.compute(ctrls, temp_control)
            _transversal1_internal_control(op, qs, ignore_, True, temp_control)
            CNXPartialAndLadder.uncompute(ctrls, cnx_ancilla, temp_control)
            discard(temp_control)


@guppy
@no_type_check
def _transversal1_start_end_internal_control[n: nat](
    op: Unitary[[qubit], None],
    qs: array[qubit, n],
    start: nat,
    end: nat,
    dag: bool @ comptime,
    ctrl: qubit,
) -> None:
    for i in range(start, end):
        with control(ctrl):
            if dag:
                with dagger:
                    op(qs[i])
            else:
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
        if n_ctrl == 1:
            _transversal1_start_end_internal_control(
                op, qs, start, end, False, ctrls[0]
            )
        else:
            temp_control = qubit()
            cnx_ancilla = CNXPartialAndLadder.compute(ctrls, temp_control)
            _transversal1_start_end_internal_control(
                op, qs, start, end, False, temp_control
            )
            CNXPartialAndLadder.uncompute(ctrls, cnx_ancilla, temp_control)
            discard(temp_control)

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat, n_ctrl: nat](  # noqa: D102
        op: Unitary[[qubit], None],
        qs: array[qubit, n],
        start: nat,
        end: nat,
        ctrls: array[qubit, n_ctrl],
    ) -> None:
        if n_ctrl == 1:
            _transversal1_start_end_internal_control(op, qs, start, end, True, ctrls[0])
        else:
            temp_control = qubit()
            cnx_ancilla = CNXPartialAndLadder.compute(ctrls, temp_control)
            _transversal1_start_end_internal_control(
                op, qs, start, end, True, temp_control
            )
            CNXPartialAndLadder.uncompute(ctrls, cnx_ancilla, temp_control)
            discard(temp_control)


@guppy(unitary=True)
@no_type_check
def transversal1_all[n: nat](op: Unitary[[qubit], None], qs: array[qubit, n]) -> None:
    """Apply a single qubit operator register-wise.

    Args:
        op: The operator to apply.
        qs: The qubit array to apply the operator to.

    """
    transversal1(op, qs, array())


@guppy
@no_type_check
def _transversal2_internal_control[n: nat, m: nat](
    op: Unitary[[qubit, qubit], None],
    qs0: array[qubit, n],
    qs1: array[qubit, n],
    ignore_: array[nat, m],
    dag: bool @ comptime,
    ctrl: qubit,
) -> None:
    for i in range(n):
        apply_gate = True
        for j in range(m):
            if i == ignore_[j]:
                apply_gate = False
        if apply_gate:
            with control(ctrl):
                if dag:
                    with dagger:
                        op(qs0[i], qs1[i])
                else:
                    op(qs0[i], qs1[i])


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
        if n_ctrl == 1:
            _transversal2_internal_control(op, qs0, qs1, ignore_, False, ctrls[0])
        else:
            temp_control = qubit()
            cnx_ancilla = CNXPartialAndLadder.compute(ctrls, temp_control)
            _transversal2_internal_control(op, qs0, qs1, ignore_, False, temp_control)
            CNXPartialAndLadder.uncompute(ctrls, cnx_ancilla, temp_control)
            discard(temp_control)

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat, m: nat, n_ctrl: nat](  # noqa: D102
        op: Unitary[[qubit, qubit], None],
        qs0: array[qubit, n],
        qs1: array[qubit, n],
        ignore_: array[nat, m],
        ctrls: array[qubit, n_ctrl],
    ) -> None:
        if n_ctrl == 1:
            _transversal2_internal_control(op, qs0, qs1, ignore_, True, ctrls[0])
        else:
            temp_control = qubit()
            cnx_ancilla = CNXPartialAndLadder.compute(ctrls, temp_control)
            _transversal2_internal_control(op, qs0, qs1, ignore_, True, temp_control)
            CNXPartialAndLadder.uncompute(ctrls, cnx_ancilla, temp_control)
            discard(temp_control)


@guppy
@no_type_check
def _transversal2_start_end_internal_control[n: nat](
    op: Unitary[[qubit, qubit], None],
    qs0: array[qubit, n],
    qs1: array[qubit, n],
    start: nat,
    end: nat,
    dag: bool @ comptime,
    ctrl: qubit,
) -> None:
    for i in range(start, end):
        with control(ctrl):
            if dag:
                with dagger:
                    op(qs0[i], qs1[i])
            else:
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
        if n_ctrl == 1:
            _transversal2_start_end_internal_control(
                op, qs0, qs1, start, end, False, ctrls[0]
            )
        else:
            temp_control = qubit()
            cnx_ancilla = CNXPartialAndLadder.compute(ctrls, temp_control)
            _transversal2_start_end_internal_control(
                op, qs0, qs1, start, end, False, temp_control
            )
            CNXPartialAndLadder.uncompute(ctrls, cnx_ancilla, temp_control)
            discard(temp_control)

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
        if n_ctrl == 1:
            _transversal2_start_end_internal_control(
                op, qs0, qs1, start, end, True, ctrls[0]
            )
        else:
            temp_control = qubit()
            cnx_ancilla = CNXPartialAndLadder.compute(ctrls, temp_control)
            _transversal2_start_end_internal_control(
                op, qs0, qs1, start, end, True, temp_control
            )
            CNXPartialAndLadder.uncompute(ctrls, cnx_ancilla, temp_control)
            discard(temp_control)


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
def _cphase(control: qubit, target: qubit, theta: angle) -> None:
    """Apply a controlled phase rotation (CU1)."""
    rz(control, theta / 2)
    crz(control, target, theta)


@guppy.unitary
class apply_phase:  # noqa: D101
    @guppy
    @no_type_check
    def __call__(q: qubit, theta: angle) -> None:
        r"""Apply a phase gate.

        Up to a global phase:
        .. math::

            \\begin{pmatrix}
            1 & 0 \\\\
            0 & e^{i\\theta}
            \\end{pmatrix}
        """
        rz(q, theta)

    @guppy
    @no_type_check
    def daggered(q: qubit, theta: angle) -> None:
        """Phase gate dagger."""
        with dagger:
            rz(q, theta)

    @guppy
    @no_type_check
    def controlled[n: nat](q: qubit, theta: angle, ctrls: array[qubit, n]) -> None:
        """Apply controlled phase gate.

        Not just equal to controlled Rz as global phase in __call__ gets converted to
        a relative phase.
        """
        if n == 1:
            _cphase(ctrls[0], q, theta)
        else:
            temp_control = qubit()
            work_ancilla = CNXPartialAndLadder.compute(ctrls, temp_control)
            _cphase(temp_control, q, theta)
            CNXPartialAndLadder.uncompute(ctrls, work_ancilla, temp_control)
            discard(temp_control)

    @guppy
    @no_type_check
    def ctrl_daggered[n: nat](q: qubit, theta: angle, ctrls: array[qubit, n]) -> None:
        """Apply control-daggered phase gate.

        Not just equal to controlled Rz as global phase gets converted to
        a relative phase.
        """
        if n == 1:
            _cphase(ctrls[0], q, -theta)
        else:
            temp_control = qubit()
            work_ancilla = CNXPartialAndLadder.compute(ctrls, temp_control)
            _cphase(temp_control, q, -theta)
            CNXPartialAndLadder.uncompute(ctrls, work_ancilla, temp_control)
            discard(temp_control)


@guppy(controllable=True)
@no_type_check
def __apply_bitstring_frozen_kernel[n: nat](
    reg: array[qubit, n], bit_array: frozenarray[bool, n]
) -> None:
    """Apply bit flips to a reg according to bit_array."""
    for i in range(n):
        if bit_array[i]:
            x(reg[i])


@guppy
@no_type_check
def __apply_bitstring_frozen_control_kernel[n: nat, n_ctrl: nat](
    reg: array[qubit, n],
    bit_array: frozenarray[bool, n],
    controls: array[qubit, n_ctrl],
) -> None:
    """Apply bitstring controlled on register, for multicontrol use a flag bit."""
    if n_ctrl == 0:
        pass
    if n_ctrl == 1:
        with control(controls[0]):
            __apply_bitstring_frozen_kernel(reg, bit_array)
    else:
        work = qubit()
        cnx(controls, work)
        with control(work):
            __apply_bitstring_frozen_kernel(reg, bit_array)
        cnx(controls, work)
        discard(work)


@guppy.unitary
class _apply_bitstring_frozen[n: nat]:
    # Apply the circuit with dagger and external-control support.

    @guppy
    @no_type_check
    def __call__(reg: array[qubit, n], bit_array: frozenarray[bool, n]) -> None:
        """Apply bit flips to a reg according to bit_array."""
        __apply_bitstring_frozen_kernel(reg, bit_array)

    @guppy
    @no_type_check
    def daggered(reg: array[qubit, n], bit_array: frozenarray[bool, n]) -> None:
        """Apply the selected circuit mode."""
        __apply_bitstring_frozen_kernel(reg, bit_array)

    @guppy
    @no_type_check
    def controlled[k: nat](
        reg: array[qubit, n], bit_array: frozenarray[bool, n], controls: array[qubit, k]
    ) -> None:
        """Apply the selected circuit mode."""
        __apply_bitstring_frozen_control_kernel(reg, bit_array, controls)

    @guppy
    @no_type_check
    def ctrl_daggered[k: nat](
        reg: array[qubit, n], bit_array: frozenarray[bool, n], controls: array[qubit, k]
    ) -> None:
        """Apply the selected circuit mode."""
        __apply_bitstring_frozen_control_kernel(reg, bit_array, controls)


@guppy(controllable=True)
@no_type_check
def __apply_bitstring_arr_kernel[n: nat](
    reg: array[qubit, n], bit_array: array[bool, n]
) -> None:
    """Apply bit flips to a reg according to bit_array."""
    for i in range(n):
        if bit_array[i]:
            x(reg[i])


@guppy
@no_type_check
def __apply_bitstring_arr_control_kernel[n: nat, n_ctrl: nat](
    reg: array[qubit, n], bit_array: array[bool, n], controls: array[qubit, n_ctrl]
) -> None:
    """Apply bitstring controlled on register, for multicontrol use a flag bit."""
    if n_ctrl == 0:
        pass
    if n_ctrl == 1:
        with control(controls[0]):
            __apply_bitstring_arr_kernel(reg, bit_array)
    else:
        work = qubit()
        cnx(controls, work)
        with control(work):
            __apply_bitstring_arr_kernel(reg, bit_array)
        cnx(controls, work)
        discard(work)


@guppy.unitary
class _apply_bitstring_arr[n: nat]:
    # Apply the circuit with dagger and external-control support.

    @guppy(controllable=True)
    @no_type_check
    def __call__(reg: array[qubit, n], bit_array: array[bool, n]) -> None:
        """Apply bit flips to a reg according to bit_array."""
        __apply_bitstring_arr_kernel(reg, bit_array)

    @guppy
    @no_type_check
    def daggered(reg: array[qubit, n], bit_array: array[bool, n]) -> None:
        """Apply the selected circuit mode."""
        __apply_bitstring_arr_kernel(reg, bit_array)

    @guppy
    @no_type_check
    def controlled[k: nat](
        reg: array[qubit, n], bit_array: array[bool, n], controls: array[qubit, k]
    ) -> None:
        """Apply the selected circuit mode."""
        __apply_bitstring_arr_control_kernel(reg, bit_array, controls)

    @guppy
    @no_type_check
    def ctrl_daggered[k: nat](
        reg: array[qubit, n], bit_array: array[bool, n], controls: array[qubit, k]
    ) -> None:
        """Apply the selected circuit mode."""
        __apply_bitstring_arr_control_kernel(reg, bit_array, controls)


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
