"""Ripple-carry Gidney modular multiplication circuits.

This module implements the textbook shift-and-add decomposition of modular
multiplication as a sequence of controlled modular additions. For each bit of the
multiplier, the multiplicand is shifted by the corresponding amount and added into
an accumulator modulo ``2**n``.

The construction follows the standard reversible circuit decomposition described in [1].

References:
    [1] Vedral, V., Barenco, A., & Ekert, A. (1996). Quantum networks for elementary
    arithmetic operations. Physical Review A, 54(1), 147.

"""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.mem import mem_swap
from guppylang.std.quantum import discard_array, qubit

from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.primitives.arithmetic.adder.adder_ripple_gidney import (
    adder_ripple_gidney_mod,
    cntrl_adder_ripple_gidney_mod,
)
from guppyalgos.utils import apply_bitstring, cswap, int_to_bits, qarray


def _negative_modular_inverse(value: int, modulus: int) -> int:
    """Return the additive inverse of ``value``'s modular inverse.

    In other words, this function returns ``-value^{-1} mod modulus``.
    """
    if value < 0:
        msg = "In-place multiplication requires a non-negative multiplier"
        raise ValueError(msg)
    if value % 2 == 0:
        msg = "In-place multiplication modulo 2**n requires an odd multiplier"
        raise ValueError(msg)
    return -pow(value, -1, modulus) % modulus


@guppy
@no_type_check
def _compute_partial_product[n: nat](
    a_reg: array[qubit, n],
    b_bit: qubit,
    partial_product: array[qubit, n],
    i: int,
) -> None:
    """Compute the partial product for the i-th bit of the multiplier.

    Computes ``b_i * (a << i)`` into the partial product register. The partial product
    register must be initialized to zero.
    """
    for j in range(n):
        if j >= i:
            temp_and_compute(b_bit, a_reg[j - i], partial_product[j])


@guppy
@no_type_check
def _uncompute_partial_product[n: nat](
    a_reg: array[qubit, n],
    b_bit: qubit,
    partial_product: array[qubit, n],
    i: int,
) -> None:
    """Uncompute the partial product for the i-th bit of the multiplier."""
    for j in range(n):
        if j >= i:
            temp_and_uncompute(b_bit, a_reg[j - i], partial_product[j])


@guppy
@no_type_check
def multiplier_ripple_gidney_mod[n: nat](
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
    product: array[qubit, n],
) -> None:
    """Add the product of ``a_reg`` and ``b_reg`` into ``product`` modulo ``2**n``.

    Performs the transformation:
    ``|a⟩|b⟩|p⟩ -> |a⟩|b⟩|p + a * b mod 2**n⟩``

    This implements the standard shift-and-add decomposition:

    ``product += b_reg[i] * (a_reg << i) mod 2**n``

    for each bit position ``i`` of the multiplier register.

    Args:
        a_reg: Multiplicand register.
        b_reg: Multiplier register. Each bit controls the corresponding shifted
            addend.
        product: Accumulator register, updated modulo ``2**n``.

    """
    for i in range(n):
        partial_product = qarray(n)
        _compute_partial_product(a_reg, b_reg[i], partial_product, i)
        adder_ripple_gidney_mod(partial_product, product)
        _uncompute_partial_product(a_reg, b_reg[i], partial_product, i)
        discard_array(partial_product)


@guppy
@no_type_check
def cntrl_multiplier_ripple_gidney_mod[n: nat](
    ctrl: qubit,
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
    product: array[qubit, n],
) -> None:
    """Conditionally multiply ``a_reg`` by ``b_reg`` into ``product`` modulo ``2**n``.

    Performs the transformation:
    ``|ctrl⟩|a⟩|b⟩|p⟩ → |ctrl⟩|a⟩|b⟩|p + ctrl * a * b mod 2**n⟩``

    This implements the standard shift-and-add decomposition:

    ``product += ctrl * b_reg[i] * (a_reg << i) mod 2**n``

    for each bit position ``i`` of the multiplier register. The global control
    ``ctrl`` gates the entire multiplication: if ``ctrl`` is ``0``, the product is
    left unchanged.

    Args:
        ctrl: Global control qubit.
        a_reg: Multiplicand register.
        b_reg: Multiplier register. Each bit controls the corresponding shifted
            addend.
        product: Accumulator register, updated modulo ``2**n``.

    """
    for i in range(n):
        partial_product = qarray(n)
        _compute_partial_product(a_reg, b_reg[i], partial_product, i)
        cntrl_adder_ripple_gidney_mod(ctrl, partial_product, product)
        _uncompute_partial_product(a_reg, b_reg[i], partial_product, i)
        discard_array(partial_product)


@guppy
@no_type_check
def multiplier_ripple_gidney_mod_in_place[n: nat](
    a_reg: array[qubit, n],
    b: int @ comptime,
) -> None:
    """Multiply ``a_reg`` by the odd classical constant ``b`` in-place.

    Performs ``|a> -> |b * a mod 2**n>``. Since multiplication modulo ``2**n``
    is invertible only for odd multipliers, ``b`` must be odd.

    Let ``M = 2**n``. The compute-swap-uncompute construction acts as
    ``(a, 0) -> (a, ba) -> (ba, a) -> (ba, a - b^{-1}(ba)) = (ba, 0) (mod M)``.

    Args:
        a_reg: Register modified in-place.
        b: Odd classical multiplier.

    Raises:
        ValueError: If ``b`` is even.

    """
    b_bits = comptime(int_to_bits(b % 2**n, n))
    negative_inverse_bits = comptime(int_to_bits(_negative_modular_inverse(b, 2**n), n))

    constant_reg = qarray(n)
    apply_bitstring(constant_reg, b_bits)
    work_reg = qarray(n)
    multiplier_ripple_gidney_mod(constant_reg, a_reg, work_reg)
    apply_bitstring(constant_reg, b_bits)
    discard_array(constant_reg)

    mem_swap(a_reg, work_reg)

    inverse_reg = qarray(n)
    apply_bitstring(inverse_reg, negative_inverse_bits)
    multiplier_ripple_gidney_mod(inverse_reg, a_reg, work_reg)
    apply_bitstring(inverse_reg, negative_inverse_bits)
    discard_array(inverse_reg)
    discard_array(work_reg)


@guppy
@no_type_check
def cntrl_multiplier_ripple_gidney_mod_in_place[n: nat](
    ctrl: qubit,
    a_reg: array[qubit, n],
    b: int @ comptime,
) -> None:
    """Conditionally multiply ``a_reg`` by the odd classical constant ``b`` in-place.

    Performs ``|ctrl>|a> -> |ctrl>|b * a mod 2**n>`` when ``ctrl`` is set and
    acts as the identity otherwise. Since multiplication modulo ``2**n`` is
    invertible only for odd multipliers, ``b`` must be odd.

    Args:
        ctrl: Control qubit.
        a_reg: Register modified in-place when ``ctrl`` is set.
        b: Odd classical multiplier.

    Raises:
        ValueError: If ``b`` is even.

    """
    b_bits = comptime(int_to_bits(b % 2**n, n))
    negative_inverse_bits = comptime(int_to_bits(_negative_modular_inverse(b, 2**n), n))

    constant_reg = qarray(n)
    apply_bitstring(constant_reg, b_bits)
    work_reg = qarray(n)
    cntrl_multiplier_ripple_gidney_mod(ctrl, constant_reg, a_reg, work_reg)
    apply_bitstring(constant_reg, b_bits)
    discard_array(constant_reg)

    cswap(ctrl, a_reg, work_reg)

    inverse_reg = qarray(n)
    apply_bitstring(inverse_reg, negative_inverse_bits)
    cntrl_multiplier_ripple_gidney_mod(ctrl, inverse_reg, a_reg, work_reg)
    apply_bitstring(inverse_reg, negative_inverse_bits)
    discard_array(inverse_reg)
    discard_array(work_reg)
