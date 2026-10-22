"""Quantum subtraction circuits."""

from typing import no_type_check

from guppylang.std.builtins import control, dagger
from guppylang import guppy
from guppylang.std.quantum import qubit, cx, x, toffoli, discard, discard_array
from guppylang.std.builtins import array, comptime, nat

from guppyalgos.primitives.arithmetic import (
    adder_ripple_cuccaro_mod,
    adder_ripple_gidney_mod,
)


from guppyalgos.utils import qarray, transversal
from guppyalgos.utils.guppy.unsafe_borrow import (
    _unsafe_array_borrow_slice,
    _unsafe_array_unborrow_slice,
)
from guppyalgos.primitives.subroutines.ladders import CXLadderLinear
from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.primitives.arithmetic.adder.adder_ripple_cuccaro import (
    _maj_umaj_bottom_carry_out_inverse_gate,
    _unmajority_gate,
    _cntrl_majority_dagger_gate,
)
from guppyalgos.primitives.arithmetic.adder.adder_ripple_gidney import (
    _g_unmajority_inverse_gate,
    _g_majority_inverse_gate,
    _gmaj_gumaj_bottom_carry_out_inverse_gate,
    _cntrl_g_majority_inverse_gate,
    _cntrl_gmaj_gumaj_bottom_carry_out_inverse_gate,
    _ccx,
)


@guppy
@no_type_check
def subtractor_ripple_cuccaro_mod[n: nat](
    a_reg: array[qubit, n], b_reg: array[qubit, n]
) -> None:
    """Apply a Cuccaro ripple-carry modular subtraction circuit.

    This circuit is the inverse of the Cuccaro ripple-carry
    adder. It computes b - a in place on b_reg:

    ``|a>|b> -> |a>|b - a mod 2^n>``

    The circuit has linear depth and requires one ancilla qubit.

    TODO: Rewrite using CNOT ladders and Toffoli ladders. This adder does not
    necessarily have linear depth, it depends on the choice of ladders.

    Reference:
    Cuccaro, Steven A., et al. "A new quantum ripple-carry addition
    circuit." arXiv preprint quant-ph/0410184 (2004).

    Args:
        ctrl (qubit): The control qubit.
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg -= a_reg mod 2^n.

    """
    with dagger:
        adder_ripple_cuccaro_mod(a_reg, b_reg)


@guppy
@no_type_check
def subtractor_ripple_cuccaro_carry_out[n: nat](
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
    borrow_out: qubit,
) -> None:
    """Apply a Cuccaro ripple-carry subtraction circuit.

    This circuit is the inverse of the Cuccaro ripple-carry
    adder. It computes b - a in place on b_reg:

    ``|a>|b>|0> -> |a>|b - a mod 2^n>|borrow_out>``

    For unsigned integers, borrow_out is flipped iff b < a.

    The circuit has linear depth and requires one ancilla qubit.

    TODO: Rewrite using CNOT ladders and Toffoli ladders. This adder does not
    necessarily have linear depth, it depends on the choice of ladders.

    Reference:
    Cuccaro, Steven A., et al. "A new quantum ripple-carry addition
    circuit." arXiv preprint quant-ph/0410184 (2004).

    Args:
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg -= a_reg mod 2^n.
        borrow_out (qubit): Clean output qubit receiving the final borrow bit.
            This is flipped iff b < a.

    """
    cx_ladder = CXLadderLinear()
    ancilla = qubit()

    if n == 1:
        cx(a_reg[0], b_reg[0])
        temp_and_compute(a_reg[0], b_reg[0], borrow_out)
        discard(ancilla)
        return

    if n == 2:
        # Boundary UMAJ†: carry-in is known to be 0 and ancilla is clean.
        cx(a_reg[0], b_reg[0])
        temp_and_compute(a_reg[0], b_reg[0], ancilla)

        _maj_umaj_bottom_carry_out_inverse_gate(
            a_reg[1], b_reg[1], ancilla, borrow_out, False
        )

        # Boundary MAJ†: uncompute the clean ancilla.
        temp_and_uncompute(a_reg[0], b_reg[0], ancilla)
        discard(ancilla)
        return

    transversal(cx, a_reg, b_reg)

    temp_and_compute(a_reg[0], b_reg[0], ancilla)

    # Forward n-1 CX ladder
    cx(a_reg[1], ancilla)
    cx_p, cx_q, cx_s = _unsafe_array_borrow_slice(a_reg, 1, comptime(n - 1), 0)
    cx_ladder.descending_dagger(cx_q)
    _unsafe_array_unborrow_slice(a_reg, cx_p, cx_q, cx_s)

    for i in range(n - 2):
        x(b_reg[i + 1])

    toffoli(ancilla, b_reg[1], a_reg[1])

    # Forward n-2 Toffoli ladder
    for i in range(n - 4, -1, -1):
        toffoli(
            a_reg[n - i - 3],
            b_reg[n - i - 2],
            a_reg[n - i - 2],
        )

    for i in range(n - 2):
        cx(a_reg[i + 1], b_reg[i + 2])
    cx(ancilla, b_reg[1])

    for i in range(n - 2):
        x(b_reg[i + 1])

    temp_and_compute(a_reg[n - 2], b_reg[n - 1], borrow_out)
    cx(a_reg[n - 1], borrow_out)

    # Reverse n-2 Toffoli ladder
    for i in range(n - 4, -1, -1):
        toffoli(a_reg[i + 1], b_reg[i + 2], a_reg[i + 2])
    toffoli(ancilla, b_reg[1], a_reg[1])

    # Reverse n-1 CX ladder
    cx_p, cx_q, cx_s = _unsafe_array_borrow_slice(a_reg, 1, comptime(n - 1), 0)
    cx_ladder.descending(cx_q)
    _unsafe_array_unborrow_slice(a_reg, cx_p, cx_q, cx_s)
    cx(a_reg[1], ancilla)

    temp_and_uncompute(a_reg[0], b_reg[0], ancilla)

    transversal(cx, a_reg, b_reg, 1, n)

    discard(ancilla)


@guppy
@no_type_check
def cntrl_subtractor_ripple_cuccaro_mod[n: nat](
    ctrl: qubit, a_reg: array[qubit, n], b_reg: array[qubit, n]
) -> None:
    """Apply a controlled Cuccaro ripple-carry modular subtraction circuit.

    This circuit is the inverse of the controlled Cuccaro ripple-carry modular
    adder. It computes b - a in place on b_reg, conditioned on the control qubit:

    ``|ctrl>|a>|b> -> |ctrl>|a>|b - ctrl * a mod 2^n>``

    The circuit has linear depth and requires one ancilla qubit.

    TODO: Rewrite using CNOT ladders and Toffoli ladders. This adder does not
    necessarily have linear depth, it depends on the choice of ladders.

    Reference:
    Cuccaro, Steven A., et al. "A new quantum ripple-carry addition
    circuit." arXiv preprint quant-ph/0410184 (2004).

    Args:
        ctrl (qubit): The control qubit.
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg -= a_reg mod 2^n.

    """
    with control(ctrl), dagger:
        adder_ripple_cuccaro_mod(a_reg, b_reg)


@guppy
@no_type_check
def cntrl_subtractor_ripple_cuccaro_carry_out[n: nat](
    ctrl: qubit,
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
    borrow_out: qubit,
) -> None:
    """Apply a controlled Cuccaro ripple-carry subtraction circuit.

    This circuit is the inverse of the Cuccaro ripple-carry
    adder. It computes b - a in place on b_reg, conditioned on the control qubit:

    ``|ctrl>|a>|b>|0> -> |ctrl>|a>|b - ctrl * a mod 2^n>|ctrl * borrow_out>``

    For unsigned integers, borrow_out is flipped iff b < a.

    The circuit has linear depth and requires one ancilla qubit.

    TODO: Rewrite using CNOT ladders and Toffoli ladders. This adder does not
    necessarily have linear depth, it depends on the choice of ladders.

    Reference:
    Cuccaro, Steven A., et al. "A new quantum ripple-carry addition
    circuit." arXiv preprint quant-ph/0410184 (2004).

    Args:
        ctrl (qubit): The control qubit.
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg -= a_reg mod 2^n.
        borrow_out (qubit): Clean output qubit receiving the final borrow bit.
            This is flipped iff b < a.

    """
    cx_ladder = CXLadderLinear()
    ancilla = qubit()

    if n == 1:
        toffoli(ctrl, a_reg[0], b_reg[0])
        temp_and_compute(a_reg[0], b_reg[0], ancilla)
        temp_and_compute(ctrl, ancilla, borrow_out)
        temp_and_uncompute(a_reg[0], b_reg[0], ancilla)
        discard(ancilla)
        return

    if n == 2:
        # Boundary UMAJ†: carry-in is known to be 0 and ancilla is clean.
        cx(a_reg[0], b_reg[0])
        temp_and_compute(a_reg[0], b_reg[0], ancilla)

        with dagger:
            _unmajority_gate(a_reg[1], b_reg[1], ancilla)
        temp_and_compute(ctrl, a_reg[1], borrow_out)
        _cntrl_majority_dagger_gate(ctrl, a_reg[1], b_reg[1], ancilla)

        # Boundary controlled-MAJ†: uncompute the clean ancilla.
        temp_and_uncompute(a_reg[0], b_reg[0], ancilla)
        toffoli(ctrl, a_reg[0], b_reg[0])
        cx(a_reg[0], b_reg[0])
        discard(ancilla)
        return

    toffoli(ctrl, a_reg[0], b_reg[0])
    transversal(cx, a_reg, b_reg, 1, n)

    temp_and_compute(a_reg[0], b_reg[0], ancilla)

    # Forward n-1 CX ladder
    cx(a_reg[1], ancilla)
    cx_p, cx_q, cx_s = _unsafe_array_borrow_slice(a_reg, 1, comptime(n - 1), 0)
    cx_ladder.descending_dagger(cx_q)
    _unsafe_array_unborrow_slice(a_reg, cx_p, cx_q, cx_s)

    # Forward n-1 Toffoli ladders
    toffoli(ctrl, ancilla, b_reg[1])
    toffoli(ancilla, b_reg[1], a_reg[1])
    for i in range(n - 3, -1, -1):
        toffoli(ctrl, a_reg[n - i - 2], b_reg[n - i - 1])
        toffoli(a_reg[n - i - 2], b_reg[n - i - 1], a_reg[n - i - 1])

    # Reverse n-1 Toffoli ladder
    temp_and_compute(ctrl, a_reg[n - 1], borrow_out)
    for i in range(n - 3, -1, -1):
        toffoli(a_reg[i + 1], b_reg[i + 2], a_reg[i + 2])
    toffoli(ancilla, b_reg[1], a_reg[1])

    # Reverse n-1 CX ladder
    cx_p, cx_q, cx_s = _unsafe_array_borrow_slice(a_reg, 1, comptime(n - 1), 0)
    cx_ladder.descending(cx_q)
    _unsafe_array_unborrow_slice(a_reg, cx_p, cx_q, cx_s)
    cx(a_reg[1], ancilla)

    temp_and_uncompute(a_reg[0], b_reg[0], ancilla)

    transversal(cx, a_reg, b_reg, 1, n)

    discard(ancilla)


@guppy
@no_type_check
def subtractor_ripple_gidney_mod[n: nat](
    a_reg: array[qubit, n], b_reg: array[qubit, n]
) -> None:
    """Apply a Gidney ripple-carry modular subtraction circuit.

    This circuit is the inverse of the Gidney ripple-carry
    adder. It computes b - a in place on b_reg:

    ``|a>|b> -> |a>|b - a mod 2^n>``

    The circuit has linear depth and requires one ancilla qubit.

    Reference:
    Gidney, C. (2018). Halving the cost of quantum addition. Quantum, 2, 74.

    Args:
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg -= a_reg mod 2^n.

    """
    with dagger:
        adder_ripple_gidney_mod(a_reg, b_reg)


@guppy
@no_type_check
def subtractor_ripple_gidney_carry_out[n: nat](
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
    borrow_out: qubit,
) -> None:
    """Apply a Gidney ripple-carry subtraction circuit.

    This circuit is the inverse of the Gidney ripple-carry
    adder. It computes b - a in place on b_reg:

    ``|a>|b>|0> -> |a>|b - a mod 2^n>|borrow_out>``

    For unsigned integers, borrow_out is flipped iff b < a.

    The circuit has linear depth and requires one ancilla qubit.

    Reference:
    Gidney, C. (2018). Halving the cost of quantum addition. Quantum, 2, 74.

    Args:
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg -= a_reg mod 2^n.
        borrow_out (qubit): Clean output qubit receiving the final borrow bit.
            This is flipped iff b < a.

    """
    if n == 1:
        cx(a_reg[0], b_reg[0])
        temp_and_compute(a_reg[0], b_reg[0], borrow_out)
        return

    anc = qarray(comptime(n - 1))

    cx(a_reg[0], b_reg[0])
    temp_and_compute(a_reg[0], b_reg[0], anc[0])

    # ------------------------------------------------------------------ #
    # Forward sweep                                                      #
    # ------------------------------------------------------------------ #
    for i in range(n - 2):
        _g_unmajority_inverse_gate(a_reg[i + 1], b_reg[i + 1], anc[i], anc[i + 1])

    _gmaj_gumaj_bottom_carry_out_inverse_gate(
        a_reg[n - 1], b_reg[n - 1], anc[n - 2], borrow_out, False
    )

    # ------------------------------------------------------------------ #
    # Reverse sweep                                                      #
    # ------------------------------------------------------------------ #

    for i in range(n - 2):
        _g_majority_inverse_gate(
            a_reg[n - 2 - i], b_reg[n - 2 - i], anc[n - 3 - i], anc[n - 2 - i]
        )

    temp_and_uncompute(a_reg[0], b_reg[0], anc[0])

    discard_array(anc)


@guppy
@no_type_check
def cntrl_subtractor_ripple_gidney_mod[n: nat](
    ctrl: qubit, a_reg: array[qubit, n], b_reg: array[qubit, n]
) -> None:
    """Apply a controlled Gidney ripple-carry modular subtraction circuit.

    This circuit is the inverse of the controlled Gidney ripple-carry modular
    adder. It computes b - a in place on b_reg, conditioned on the control qubit:

    ``|ctrl>|a>|b> -> |ctrl>|a>|b - ctrl * a mod 2^n>``

    Reference:
    Gidney, C. (2018). Halving the cost of quantum addition. Quantum, 2, 74.

    Args:
        ctrl (qubit): The control qubit.
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg -= a_reg mod 2^n.

    """
    with control(ctrl), dagger:
        adder_ripple_gidney_mod(a_reg, b_reg)


@guppy
@no_type_check
def cntrl_subtractor_ripple_gidney_carry_out[n: nat](
    ctrl: qubit,
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
    borrow_out: qubit,
) -> None:
    """Apply a controlled Gidney ripple-carry subtraction circuit.

    This circuit is the inverse of the Gidney ripple-carry
    adder. It computes b - a in place on b_reg, conditioned on the control qubit:

    ``|ctrl>|a>|b>|0> -> |ctrl>|a>|b - ctrl * a mod 2^n>|ctrl * borrow_out>``

    For unsigned integers, borrow_out is flipped iff b < a.

    Reference:
    Gidney, C. (2018). Halving the cost of quantum addition. Quantum, 2, 74.

    Args:
        ctrl (qubit): The control qubit.
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg -= a_reg mod 2^n.
        borrow_out (qubit): Clean output qubit receiving the final borrow bit.
            This is flipped iff b < a.

    """
    if n == 1:
        ancilla = qubit()
        toffoli(ctrl, a_reg[0], b_reg[0])
        temp_and_compute(a_reg[0], b_reg[0], ancilla)
        temp_and_compute(ctrl, ancilla, borrow_out)
        temp_and_uncompute(a_reg[0], b_reg[0], ancilla)
        discard(ancilla)
        return

    anc = qarray(comptime(n - 1))

    _ccx(ctrl, a_reg[0], b_reg[0])
    temp_and_compute(a_reg[0], b_reg[0], anc[0])

    # ------------------------------------------------------------------ #
    # Forward sweep                                                      #
    # ------------------------------------------------------------------ #
    for i in range(n - 2):
        _g_unmajority_inverse_gate(a_reg[i + 1], b_reg[i + 1], anc[i], anc[i + 1])

    _cntrl_gmaj_gumaj_bottom_carry_out_inverse_gate(
        ctrl, a_reg[n - 1], b_reg[n - 1], anc[n - 2], borrow_out, False
    )

    # ------------------------------------------------------------------ #
    # Reverse sweep                                                      #
    # ------------------------------------------------------------------ #

    for i in range(n - 2):
        _cntrl_g_majority_inverse_gate(
            ctrl, a_reg[n - 2 - i], b_reg[n - 2 - i], anc[n - 3 - i], anc[n - 2 - i]
        )

    temp_and_uncompute(a_reg[0], b_reg[0], anc[0])

    discard_array(anc)
