"""Quantum subtraction circuits."""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import qubit

from guppyalgos.primitives.arithmetic import (
    adder_ripple_cuccaro_mod_dagger,
    adder_ripple_gidney_mod_dagger,
    cntrl_adder_ripple_cuccaro_mod_dagger,
    cntrl_adder_ripple_gidney_mod_dagger,
)
from guppyalgos.primitives.arithmetic.adder.adder_ripple_cuccaro import (
    _adder_ripple_cuccaro_carry_out_dagger_impl,
    _cntrl_adder_ripple_cuccaro_carry_out_dagger_impl,
)
from guppyalgos.primitives.arithmetic.adder.adder_ripple_gidney import (
    _adder_ripple_gidney_carry_out_dagger_impl,
    _cntrl_adder_ripple_gidney_carry_out_dagger_impl,
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
    return adder_ripple_cuccaro_mod_dagger(a_reg, b_reg)


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
    return _adder_ripple_cuccaro_carry_out_dagger_impl(a_reg, b_reg, borrow_out, False)


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
    return cntrl_adder_ripple_cuccaro_mod_dagger(ctrl, a_reg, b_reg)


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
    return _cntrl_adder_ripple_cuccaro_carry_out_dagger_impl(
        ctrl, a_reg, b_reg, borrow_out, False
    )


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
    return adder_ripple_gidney_mod_dagger(a_reg, b_reg)


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
    return _adder_ripple_gidney_carry_out_dagger_impl(a_reg, b_reg, borrow_out, False)


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
    return cntrl_adder_ripple_gidney_mod_dagger(ctrl, a_reg, b_reg)


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
    return _cntrl_adder_ripple_gidney_carry_out_dagger_impl(
        ctrl, a_reg, b_reg, borrow_out, False
    )
