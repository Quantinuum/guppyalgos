"""Ripple-carry Gidney modular multiplication circuits.

This module implements the textbook shift-and-add decomposition of modular
multiplication as a sequence of controlled modular additions. For each bit of the
multiplier, the multiplicand is shifted by the corresponding amount and added into
an accumulator modulo ``2**n``.

The construction follows the standard reversible circuit decomposition described in [1]
and uses the ripple-carry adder from [2].

References:
    [1] Rines, R., & Chuang, I. (2018). High performance quantum modular multipliers.
    arXiv preprint arXiv:1801.01081.
    [2] Gidney, C. (2018). Halving the cost of quantum addition. Quantum, 2, 74.

"""

from typing import no_type_check

from guppylang.std.builtins import control
from guppylang import guppy
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.mem import mem_swap
from guppylang.std.quantum import discard, qubit

from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.primitives.arithmetic.adder.adder_ripple_gidney import (
    adder_ripple_gidney_mod,
)
from guppyalgos.utils import apply_bitstring, cswap, int_to_bits, qarray
from guppyalgos.utils.guppy.unsafe_borrow import (
    _unsafe_array_borrow_slice,
    _unsafe_array_unborrow_slice,
)


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


def _require_positive_width(n: int) -> None:
    """Reject zero-width multiplier registers during Guppy compilation."""
    if n < 1:
        msg = "n must be positive"
        raise ValueError(msg)


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

    Raises:
        ValueError: If ``n`` is less than one.

    """
    comptime(_require_positive_width(n))
    with control(b_reg[0]):
        adder_ripple_gidney_mod(a_reg, product)
    if n == 1:
        return

    a_prefix, a_mid, a_suffix = _unsafe_array_borrow_slice(
        a_reg, 0, comptime(max(1, n - 1)), comptime(min(1, n - 1))
    )
    b_prefix, b_mid, b_suffix = _unsafe_array_borrow_slice(
        b_reg, comptime(min(1, n - 1)), comptime(max(1, n - 1)), 0
    )
    p_prefix, p_mid, p_suffix = _unsafe_array_borrow_slice(
        product, comptime(min(1, n - 1)), comptime(max(1, n - 1)), 0
    )
    multiplier_ripple_gidney_mod(a_mid, b_mid, p_mid)
    _unsafe_array_unborrow_slice(a_reg, a_prefix, a_mid, a_suffix)
    _unsafe_array_unborrow_slice(b_reg, b_prefix, b_mid, b_suffix)
    _unsafe_array_unborrow_slice(product, p_prefix, p_mid, p_suffix)


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

    Raises:
        ValueError: If ``n`` is less than one.

    """
    comptime(_require_positive_width(n))
    temp_control = qubit()
    temp_and_compute(ctrl, b_reg[0], temp_control)
    with control(temp_control):
        adder_ripple_gidney_mod(a_reg, product)
    temp_and_uncompute(ctrl, b_reg[0], temp_control)
    discard(temp_control)
    if n == 1:
        return

    a_prefix, a_mid, a_suffix = _unsafe_array_borrow_slice(
        a_reg, 0, comptime(max(1, n - 1)), comptime(min(1, n - 1))
    )
    b_prefix, b_mid, b_suffix = _unsafe_array_borrow_slice(
        b_reg, comptime(min(1, n - 1)), comptime(max(1, n - 1)), 0
    )
    p_prefix, p_mid, p_suffix = _unsafe_array_borrow_slice(
        product, comptime(min(1, n - 1)), comptime(max(1, n - 1)), 0
    )
    cntrl_multiplier_ripple_gidney_mod(ctrl, a_mid, b_mid, p_mid)
    _unsafe_array_unborrow_slice(a_reg, a_prefix, a_mid, a_suffix)
    _unsafe_array_unborrow_slice(b_reg, b_prefix, b_mid, b_suffix)
    _unsafe_array_unborrow_slice(product, p_prefix, p_mid, p_suffix)


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
        ValueError: If ``n`` is less than one or ``b`` is negative or even.

    """
    comptime(_require_positive_width(n))
    b_bits = comptime(int_to_bits(b % 2**n, n))
    negative_inverse_bits = comptime(int_to_bits(_negative_modular_inverse(b, 2**n), n))

    constant_reg = qarray(n)
    apply_bitstring(constant_reg, b_bits)
    work_reg = qarray(n)
    multiplier_ripple_gidney_mod(constant_reg, a_reg, work_reg)
    apply_bitstring(constant_reg, b_bits)
    for q in constant_reg:
        discard(q)

    mem_swap(a_reg, work_reg)

    inverse_reg = qarray(n)
    apply_bitstring(inverse_reg, negative_inverse_bits)
    multiplier_ripple_gidney_mod(inverse_reg, a_reg, work_reg)
    apply_bitstring(inverse_reg, negative_inverse_bits)
    for q in inverse_reg:
        discard(q)
    for q in work_reg:
        discard(q)


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
        ValueError: If ``n`` is less than one or ``b`` is negative or even.

    """
    comptime(_require_positive_width(n))
    b_bits = comptime(int_to_bits(b % 2**n, n))
    negative_inverse_bits = comptime(int_to_bits(_negative_modular_inverse(b, 2**n), n))

    constant_reg = qarray(n)
    apply_bitstring(constant_reg, b_bits)
    work_reg = qarray(n)
    cntrl_multiplier_ripple_gidney_mod(ctrl, constant_reg, a_reg, work_reg)
    apply_bitstring(constant_reg, b_bits)
    for q in constant_reg:
        discard(q)

    cswap(ctrl, a_reg, work_reg)

    inverse_reg = qarray(n)
    apply_bitstring(inverse_reg, negative_inverse_bits)
    cntrl_multiplier_ripple_gidney_mod(ctrl, inverse_reg, a_reg, work_reg)
    apply_bitstring(inverse_reg, negative_inverse_bits)
    for q in inverse_reg:
        discard(q)
    for q in work_reg:
        discard(q)
