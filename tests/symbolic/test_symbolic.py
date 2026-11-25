"""Verify the classical reversible circuits of guppyalgos against their specs."""

import gc
import inspect
import linecache
import textwrap
from contextlib import nullcontext
from functools import partial
from typing import no_type_check

import pytest
from guppylang import comptime, guppy
from guppylang.std.builtins import Function, array, control, dagger, nat
from guppylang.std.quantum import cx, discard, qubit, toffoli

from guppyalgos.algorithms.select.accumulator import (
    accumulator_unary_iteration,
    guarded_accumulator_unary_iteration,
)
from guppyalgos.algorithms.select.qrom import qrom_unary_iteration
from guppyalgos.algorithms.select.qrom.controlled_qrom_unary_iteration import (
    cntrl_qrom_unary_iteration,
)
from guppyalgos.algorithms.select.selectswap import selectswap
from guppyalgos.primitives.arithmetic import (
    adder_ripple_cuccaro_carry_out,
    adder_ripple_cuccaro_carry_out_dagger,
    adder_ripple_cuccaro_mod,
    adder_ripple_gidney_carry_out,
    adder_ripple_gidney_carry_out_dagger,
    adder_ripple_gidney_mod,
    cntrl_adder_ripple_cuccaro_carry_out,
    cntrl_adder_ripple_cuccaro_carry_out_dagger,
    cntrl_adder_ripple_cuccaro_mod,
    cntrl_adder_ripple_gidney_carry_out,
    cntrl_adder_ripple_gidney_carry_out_dagger,
    cntrl_adder_ripple_gidney_mod,
    cntrl_multiplier_ripple_gidney_mod,
    cntrl_multiplier_ripple_gidney_mod_in_place,
    cntrl_subtractor_ripple_cuccaro_carry_out,
    cntrl_subtractor_ripple_cuccaro_mod,
    cntrl_subtractor_ripple_gidney_carry_out,
    cntrl_subtractor_ripple_gidney_mod,
    exponentiator_ripple_gidney_mod,
    multiplier_ripple_gidney_mod,
    multiplier_ripple_gidney_mod_in_place,
    subtractor_ripple_cuccaro_carry_out,
    subtractor_ripple_cuccaro_mod,
    subtractor_ripple_gidney_carry_out,
    subtractor_ripple_gidney_mod,
)
from guppyalgos.primitives.arithmetic.adder.bit_adders import full_adder, half_adder
from guppyalgos.primitives.arithmetic.comparator import (
    comparator_ripple_cuccaro,
    comparator_vandaele,
    partial_comparator_ripple_cuccaro,
)
from guppyalgos.primitives.arithmetic.hamming_weight import (
    hamming_weight_func,
    hamming_weight_func_inv,
)
from guppyalgos.primitives.arithmetic.incrementer.incrementer_cca import (
    cca_incrementer,
    required_clean_ancillas,
)
from guppyalgos.primitives.arithmetic.incrementer.linear_depth_incrementer import (
    linear_depth_incrementer,
)
from guppyalgos.primitives.gate_decompositions.and_op import (
    and_pattern_controlled,
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.primitives.gate_decompositions.cnx.cnx import (
    cnx,
    cnx_single_ancilla,
    cnx_toffoli_ladder,
)
from guppyalgos.primitives.gate_decompositions.cnx.cnx_cca import (
    cnx_cca_logdepth,
    cnx_cca_logdepth_dirty,
)
from guppyalgos.primitives.subroutines.equality.equality_test import equality_test
from guppyalgos.primitives.subroutines.fanout.fanout_basic import fanout_basic
from guppyalgos.primitives.subroutines.fanout.fanout_helpers import fanout_from_data
from guppyalgos.primitives.subroutines.fanout.fanout_log import fanout_log
from guppyalgos.primitives.subroutines.ladders import (
    CXLadderLinear,
    CXLadderLog,
    ToffoliLadderLinear,
    ToffoliLadderLog,
    ccx_v_chain,
    ccx_v_chain_logdepth,
    cnx_ladder_logdepth,
    cnx_ladder_logdepth_num_ancilla,
    log_toffoli_ladder_num_ancilla,
)
from guppyalgos.primitives.subroutines.ladders.cx_ladder import (
    _cx_ladder_apply_from_inds,
    log_cx_ladder_indices,
)
from guppyalgos.primitives.subroutines.swap.swapup import (
    swapup_linear,
    swapup_linear_register,
)
from guppyalgos.primitives.utilities.gates import swap
from tests.symbolic.specs import (
    add,
    add_carry,
    add_product,
    cnx_ladder,
    cx_ladder,
    dec,
    decode,
    equal,
    fanout,
    fanout_mask,
    flip_prefix,
    full_add,
    full_add_inverse,
    half_add,
    half_add_inverse,
    inc,
    less,
    lookup,
    masked_fanout,
    mul_const,
    multi_controlled_x,
    multi_controlled_x_dirty,
    multiply_by_power,
    on_low_bits,
    qrom_table,
    same,
    select_swap,
    selectswap_entries,
    sub,
    sub_carry,
    swap_if_both,
    swap_up,
    toffoli_ladder,
    toffoli_v_chain,
    with_control,
    xor_hamming_weight,
)
from tests.symbolic.verifier import compile_together, verify

SIZES = [*range(1, 11), 15, 20, 25, 30]
ODD_SIZES = [*range(1, 12, 2), 15, 21, 25, 29]
SMALL_SIZES = range(1, 11)


# wrappers of library functions that wrap() cannot call as they are


@guppy
@no_type_check
def daggered_half_adder(a: qubit, b: qubit, c_out: qubit) -> None:
    """Apply the dagger of the half adder."""
    with dagger:
        half_adder(a, b, c_out)


@guppy
@no_type_check
def daggered_full_adder(c_in: qubit, a: qubit, b: qubit, c_out: qubit) -> None:
    """Apply the dagger of the full adder."""
    with dagger:
        full_adder(c_in, a, b, c_out)


def adder_then_dagger(adder, dagger):
    """Return ``adder`` then ``dagger``, on a carry qubit that must come back clean."""

    def f[n: nat](a: array[qubit, n], b: array[qubit, n], carry: qubit) -> None:
        adder(a, b, carry)
        dagger(a, b, carry)

    f.__name__ = dagger.wrapped.name
    return guppy(f)


def cntrl_adder_then_dagger(adder, dagger):
    """Return ``adder_then_dagger`` for a controlled adder and its dagger."""

    def f[n: nat](
        c: qubit, a: array[qubit, n], b: array[qubit, n], carry: qubit
    ) -> None:
        adder(c, a, b, carry)
        dagger(c, a, b, carry)

    f.__name__ = dagger.wrapped.name
    return guppy(f)


def multiplier_in_place_by(k):
    """Return the in-place multiplier by the constant ``k``."""

    def f[n: nat](a: array[qubit, n]) -> None:
        multiplier_ripple_gidney_mod_in_place(a, k)

    f.__name__ = f"multiplier_in_place_by_{k}"
    return guppy(f)


def cntrl_multiplier_in_place_by(k):
    """Return the controlled in-place multiplier by the constant ``k``."""

    def f[n: nat](c: qubit, a: array[qubit, n]) -> None:
        cntrl_multiplier_ripple_gidney_mod_in_place(c, a, k)

    f.__name__ = f"cntrl_multiplier_in_place_by_{k}"
    return guppy(f)


def exponentiator(base):
    """Return the exponentiator of the constant ``base``."""

    def f[m: nat, n: nat](e: array[qubit, m], a: array[qubit, n]) -> None:
        exponentiator_ripple_gidney_mod(e, a, base)

    f.__name__ = f"exponentiator_base_{base}"
    return guppy(f)


@guppy
@no_type_check
def partial_comparator[n: nat](
    a: array[qubit, n], b: array[qubit, n], t: qubit
) -> None:
    """Flip ``t`` by ``a < b``, read between the forward half and the inverse one."""
    target, ancilla = qubit(), qubit()
    partial_comparator_ripple_cuccaro(a, b, target, ancilla, False, True)
    cx(target, t)
    partial_comparator_ripple_cuccaro(a, b, target, ancilla, True, True)
    discard(target)
    discard(ancilla)


@guppy
@no_type_check
def linear_depth_decrementer[n: nat](q: array[qubit, n]) -> None:
    """Decrement all bits with the unitary incrementer's adjoint."""
    with dagger:
        linear_depth_incrementer(q)


@guppy
@no_type_check
def cca_decrementer[n: nat](q: array[qubit, n]) -> None:
    """Decrement with the CCA incrementer's adjoint."""
    with dagger:
        cca_incrementer(q)


@guppy
@no_type_check
def cntrl_linear_depth_incrementer[n: nat](c: qubit, q: array[qubit, n]) -> None:
    """Increment all bits when the control is set."""
    with control(c):
        linear_depth_incrementer(q)


@guppy
@no_type_check
def cntrl_cca_incrementer[n: nat](c: qubit, q: array[qubit, n]) -> None:
    """Apply the CCA incrementer when the control is set."""
    with control(c):
        cca_incrementer(q)


@guppy
@no_type_check
def cntrl_linear_depth_decrementer[n: nat](c: qubit, q: array[qubit, n]) -> None:
    """Decrement all bits when the control is set."""
    with control(c), dagger:
        linear_depth_incrementer(q)


@guppy
@no_type_check
def cntrl_cca_decrementer[n: nat](c: qubit, q: array[qubit, n]) -> None:
    """Apply the CCA incrementer's adjoint when the control is set."""
    with control(c), dagger:
        cca_incrementer(q)


@guppy
@no_type_check
def linear_depth_incrementer_bounded[n: nat](q: array[qubit, n]) -> None:
    """Increment with the register size as the value bound."""
    linear_depth_incrementer(q, n)


@guppy
@no_type_check
def linear_depth_decrementer_bounded[n: nat](q: array[qubit, n]) -> None:
    """Decrement with the register size as the value bound."""
    with dagger:
        linear_depth_incrementer(q, n)


@guppy
@no_type_check
def cntrl_linear_depth_incrementer_bounded[n: nat](
    c: qubit, q: array[qubit, n]
) -> None:
    """Increment with the register size as the value bound when controlled."""
    with control(c):
        linear_depth_incrementer(q, n)


@guppy
@no_type_check
def cntrl_linear_depth_decrementer_bounded[n: nat](
    c: qubit, q: array[qubit, n]
) -> None:
    """Decrement with the register size as the value bound when controlled."""
    with control(c), dagger:
        linear_depth_incrementer(q, n)


@guppy
@no_type_check
def equality[n: nat](lhs: array[qubit, n], rhs: array[qubit, n], flag: qubit) -> None:
    """Flip ``flag`` if ``lhs`` and ``rhs`` are equal, with an exact CnX."""
    equality_test(cnx_single_ancilla[n], lhs, rhs, flag)


@guppy(unitary=True)
@no_type_check
def ccswap_qubits(c0: qubit, c1: qubit, a: qubit, b: qubit) -> None:
    """Swap two qubits under two controls."""
    with control(c0, c1):
        swap(a, b)


@guppy(unitary=True)
@no_type_check
def ccswap_registers[n: nat](
    c0: qubit, c1: qubit, a: array[qubit, n], b: array[qubit, n]
) -> None:
    """Swap two registers under two controls."""
    with control(c0, c1):
        swap(a, b)


def decoder(compute):
    """Return a decoder from 2 qubits to 4, with indexed ANDs that apply these ops."""

    @no_type_check
    def f(index: array[qubit, 2], out: array[qubit, 4]) -> None:
        for i in range(4):
            low, high = i % 2 == 1, i >= 2
            target = qubit()
            and_pattern_controlled(target, index[0], low, index[1], high, compute)
            cx(target, out[i])
            with dagger:
                and_pattern_controlled(target, index[0], low, index[1], high, compute)
            discard(target)

    f.__name__ = f"temp_and_index_{compute.wrapped.name}"
    return guppy(f)


def method(struct, name):
    """Return a method of a guppy struct as a function of one qubit array."""
    bound = getattr(struct.wrapped.python_class, name)

    def f[n: nat](q: array[qubit, n]) -> None:
        bound(struct(), q)

    f.__name__ = f"{struct.wrapped.name}.{name}"
    return guppy(f)


def daggered_cx_ladder(n):
    """Return the dagger of the log depth CX ladder."""

    @guppy(unitary=True)
    @no_type_check
    def f(q: array[qubit, n]) -> None:
        with dagger:
            _cx_ladder_apply_from_inds(q, comptime(log_cx_ladder_indices(n)))

    return f


def controlled_cx_ladder(n):
    """Return the log depth CX ladder under a control."""

    @guppy
    @no_type_check
    def f(c: qubit, q: array[qubit, n]) -> None:
        with control(c):
            _cx_ladder_apply_from_inds(q, comptime(log_cx_ladder_indices(n)))

    return f


def controlled_daggered_cx_ladder(n):
    """Return the dagger of the log depth CX ladder under a control."""

    @guppy
    @no_type_check
    def f(c: qubit, q: array[qubit, n]) -> None:
        with control(c), dagger:
            _cx_ladder_apply_from_inds(q, comptime(log_cx_ladder_indices(n)))

    return f


# components that the library's factories build from classical data


def hamming_weight(n):
    """Return a circuit that XORs the Hamming weight of ``m`` into ``w``."""
    compute, uncompute = hamming_weight_func(n), hamming_weight_func_inv(n)

    @guppy
    def f(m: array[qubit, n], w: array[qubit, n]) -> None:
        weight, junk = compute(m)
        for i in range(len(weight)):
            cx(weight[i], w[i])
        uncompute(m, weight, junk)

    return f


def fanout_from_mask(n):
    """Return a fanout onto the qubits that ``fanout_mask(n)`` selects."""
    return fanout_from_data(fanout_mask(n), fanout_basic)


def selectswap_on_entries(n):
    """Return SelectSWAP on n registers using ``selectswap_entries(n, 3)``."""
    return selectswap(selectswap_entries(n, 3), n)


def qrom_on_table(qrom, compute, uncompute):
    """Return a factory of ``qrom`` on ``qrom_table(n)``, using the given AND ops."""

    def f(n):
        return qrom(qrom_table(n), comp_and_op=compute, uncomp_and_op=uncompute)

    f.__name__ = f"{qrom.__name__}_{compute.wrapped.name}"
    return f


def cx_onto(k, n):
    """Return an op that flips ``state[k]`` if its control is set."""

    @guppy
    def op(control: qubit, state: array[qubit, n]) -> None:
        cx(control, state[k])

    return op


def accumulator(run, compute, uncompute):
    """Return a factory of ``run`` over the ops ``cx_onto(k, n)``, with these ANDs."""

    def f(n):
        width = (n - 1).bit_length()

        @guppy.comptime
        def flips() -> array[Function[[qubit, array[qubit, n]], None], n]:
            return [cx_onto(k, n) for k in range(n)]

        @guppy
        def g(index: array[qubit, width], state: array[qubit, n]) -> None:
            run(flips(), compute, uncompute, index, state)

        return g

    f.__name__ = f"{run.wrapped.name}_{compute.wrapped.name}"
    return f


def accumulator_ancillas(n):
    """Return the work qubits and the accumulator qubit of an accumulator of n items."""
    return (n - 1).bit_length() if n > 2 else 0


def guarded_accumulator_ancillas(n):
    """Return the guarded accumulator's ancillas: one more unless n = 2^k or 2^k - 1."""
    if n & (n - 1) == 0 or n & (n + 1) == 0:
        return accumulator_ancillas(n)
    return accumulator_ancillas(n) + 1


# (component, shapes, spec, sizes, zero, ancillas, modifiers), as README.md explains
CASES: list[tuple] = [
    (half_adder, "qqq", half_add, [1], (), 0),
    (daggered_half_adder, "qqq", half_add_inverse, [1], (), 0),
    (full_adder, "qqqq", full_add, [1], (), 0),
    (daggered_full_adder, "qqqq", full_add_inverse, [1], (), 0),
    (adder_ripple_cuccaro_mod, "nn", add, SIZES, (), 1),
    (adder_ripple_gidney_mod, "nn", add, SIZES, (), lambda n: n - 1),
    (adder_ripple_cuccaro_carry_out, "nnq", add_carry, SIZES, ("c",), 1),
    (adder_ripple_gidney_carry_out, "nnq", add_carry, SIZES, ("c",), lambda n: n - 1),
    (cntrl_adder_ripple_cuccaro_mod, "qnn", with_control(add), SIZES, (), 1),
    (cntrl_adder_ripple_gidney_mod, "qnn", with_control(add), SIZES, (), lambda n: n),
    (
        cntrl_adder_ripple_cuccaro_carry_out,
        "qnnq",
        with_control(add_carry),
        SIZES,
        ("d",),
        1,
    ),
    (
        cntrl_adder_ripple_gidney_carry_out,
        "qnnq",
        with_control(add_carry),
        SIZES,
        ("d",),
        lambda n: n,
    ),
    (
        adder_then_dagger(
            adder_ripple_cuccaro_carry_out, adder_ripple_cuccaro_carry_out_dagger
        ),
        "nnq",
        same,
        SIZES,
        ("c",),
        1,
    ),
    (
        adder_then_dagger(
            adder_ripple_gidney_carry_out, adder_ripple_gidney_carry_out_dagger
        ),
        "nnq",
        same,
        SIZES,
        ("c",),
        lambda n: n - 1,
    ),
    (
        cntrl_adder_then_dagger(
            cntrl_adder_ripple_cuccaro_carry_out,
            cntrl_adder_ripple_cuccaro_carry_out_dagger,
        ),
        "qnnq",
        same,
        SIZES,
        ("d",),
        1,
    ),
    (
        cntrl_adder_then_dagger(
            cntrl_adder_ripple_gidney_carry_out,
            cntrl_adder_ripple_gidney_carry_out_dagger,
        ),
        "qnnq",
        same,
        SIZES,
        ("d",),
        lambda n: n,
    ),
    (subtractor_ripple_cuccaro_mod, "nn", sub, SIZES, (), 1),
    (subtractor_ripple_gidney_mod, "nn", sub, SIZES, (), lambda n: n - 1),
    (subtractor_ripple_cuccaro_carry_out, "nnq", sub_carry, SIZES, ("c",), 1),
    (
        subtractor_ripple_gidney_carry_out,
        "nnq",
        sub_carry,
        SIZES,
        ("c",),
        lambda n: n - 1,
    ),
    (cntrl_subtractor_ripple_cuccaro_mod, "qnn", with_control(sub), SIZES, (), 1),
    (
        cntrl_subtractor_ripple_gidney_mod,
        "qnn",
        with_control(sub),
        SIZES,
        (),
        lambda n: n,
    ),
    (
        cntrl_subtractor_ripple_cuccaro_carry_out,
        "qnnq",
        with_control(sub_carry),
        SIZES,
        ("d",),
        1,
    ),
    (
        cntrl_subtractor_ripple_gidney_carry_out,
        "qnnq",
        with_control(sub_carry),
        SIZES,
        ("d",),
        lambda n: n,
    ),
    (multiplier_ripple_gidney_mod, "nnn", add_product, SMALL_SIZES, (), lambda n: n),
    (
        cntrl_multiplier_ripple_gidney_mod,
        "qnnn",
        with_control(add_product),
        SMALL_SIZES,
        (),
        lambda n: n + 1,
    ),
    *[
        (multiplier(k), shape, partial(spec, k=k), SMALL_SIZES, (), ancillas)
        for multiplier, shape, spec, ancillas in [
            (multiplier_in_place_by, "n", mul_const, lambda n: 3 * n),
            (
                cntrl_multiplier_in_place_by,
                "qn",
                with_control(mul_const),
                lambda n: 3 * n + 1,
            ),
        ]
        for k in (3, 5, 7)
    ],
    (
        exponentiator(3),
        "nn",
        partial(multiply_by_power, base=3),
        range(1, 9),
        (),
        lambda n: 3 * n + 1,
    ),
    (
        exponentiator(5),
        "in",
        partial(multiply_by_power, base=5),
        range(2, 11),
        (),
        lambda n: 3 * n + 1,
    ),
    (
        exponentiator(7),
        "ni",
        partial(multiply_by_power, base=7),
        range(1, 17),
        (),
        None,
    ),
    (hamming_weight, "nn", xor_hamming_weight, SIZES[1:], (), lambda n: n),
    (comparator_ripple_cuccaro, "nnq", less, SIZES, (), 1),
    (partial_comparator, "nnq", less, SIZES, (), 2),  # its ancilla and its target
    (comparator_vandaele, "nnq", less, SIZES, (), 0),
    (linear_depth_incrementer, "n", inc, SIZES, (), lambda n: max(n - 2, 0)),
    (cca_incrementer, "n", inc, SIZES, (), required_clean_ancillas),
    (linear_depth_decrementer, "n", dec, SIZES, (), lambda n: max(n - 2, 0)),
    (cca_decrementer, "n", dec, SIZES, (), required_clean_ancillas),
    (
        cntrl_linear_depth_incrementer,
        "qn",
        with_control(inc),
        SIZES,
        (),
        lambda n: n - 1,
    ),
    (
        cntrl_cca_incrementer,
        "qn",
        with_control(inc),
        SIZES,
        (),
        lambda n: required_clean_ancillas(n + 1) if n > 2 else 0,
    ),
    (
        cntrl_linear_depth_decrementer,
        "qn",
        with_control(dec),
        SIZES,
        (),
        lambda n: n - 1,
    ),
    (
        cntrl_cca_decrementer,
        "qn",
        with_control(dec),
        SIZES,
        (),
        lambda n: required_clean_ancillas(n + 1) if n > 2 else 0,
    ),
    (
        linear_depth_incrementer_bounded,
        "n",
        on_low_bits(inc),
        SIZES,
        (),
        None,
    ),
    (
        linear_depth_decrementer_bounded,
        "n",
        on_low_bits(dec),
        SIZES,
        (),
        None,
    ),
    (
        cntrl_linear_depth_incrementer_bounded,
        "qn",
        on_low_bits(with_control(inc)),
        SIZES,
        (),
        None,
    ),
    (
        cntrl_linear_depth_decrementer_bounded,
        "qn",
        on_low_bits(with_control(dec)),
        SIZES,
        (),
        None,
    ),
    (cnx, "nq", multi_controlled_x, SIZES, (), lambda n: (n - 1) // 2, True),
    (
        cnx_single_ancilla,
        "nq",
        multi_controlled_x,
        SIZES,
        (),
        lambda n: int(n >= 3),
        True,
    ),
    (
        cnx_toffoli_ladder,
        "nq",
        multi_controlled_x,
        SIZES,
        (),
        lambda n: max(n - 2, 0),
        True,
    ),
    (
        cnx_cca_logdepth,
        "nq",
        multi_controlled_x,
        SIZES,
        (),
        lambda n: 0 if n < 3 else 1 if n < 6 else 2,
        True,
    ),
    (cnx_cca_logdepth_dirty, "nqqq", multi_controlled_x_dirty, SIZES, (), 0),
    (equality, "nnq", equal, SIZES, (), lambda n: int(n >= 3)),
    (fanout_basic, "qn", fanout, SIZES, (), 0, True),
    (fanout_log, "qn", fanout, SIZES, (), 0, True),
    (fanout_from_mask, "qn", masked_fanout, SIZES, (), None),
    (ccx_v_chain, "nnq", toffoli_v_chain, SIZES, (), 0),
    (ccx_v_chain_logdepth, "nnq", toffoli_v_chain, SIZES, (), 0),
    (swapup_linear, "in", swap_up, SIZES[1:], (), 0),
    (
        swapup_linear_register,
        ("i", "array[array[qubit, 3], {n}]"),
        swap_up,
        SIZES[1:],
        (),
        0,
    ),
    (
        selectswap_on_entries,
        ("i", "i", "array[array[qubit, 3], {n}]"),
        select_swap,
        SIZES[2:],
        (),
        None,
    ),
    (ccswap_qubits, "qqqq", swap_if_both, [1], (), 1, True),
    (ccswap_registers, "qqnn", swap_if_both, SIZES[1:], (), 1, True),
]
CASES += [  # the unary iterations, with each pair of AND ops
    case
    for ops in [(toffoli, toffoli), (temp_and_compute, temp_and_uncompute)]
    for case in [
        (decoder(ops[0]), "in", decode, [4], (), 1),
        (
            qrom_on_table(qrom_unary_iteration, *ops),
            "in",
            lookup,
            SIZES[1:],
            (),
            lambda n: (n - 1).bit_length() - 1,
        ),
        (
            qrom_on_table(cntrl_qrom_unary_iteration, *ops),
            "qin",
            with_control(lookup),
            SIZES[1:],
            (),
            lambda n: (n - 1).bit_length(),
        ),
        (
            accumulator(accumulator_unary_iteration, *ops),
            "in",
            flip_prefix,
            [2, 4, 8, 16, 32],
            (),
            accumulator_ancillas,
        ),
        (
            accumulator(guarded_accumulator_unary_iteration, *ops),
            "in",
            flip_prefix,
            SIZES[1:],
            (),
            guarded_accumulator_ancillas,
        ),
    ]
]
LADDERS = [  # (struct, spec of its ascending ladder, sizes, ancillas)
    (CXLadderLinear, cx_ladder, SIZES, 0),
    (CXLadderLog, cx_ladder, SIZES, 0),
    (ToffoliLadderLinear, toffoli_ladder, ODD_SIZES, 0),
    # through its _with_cca methods
    (ToffoliLadderLog, toffoli_ladder, ODD_SIZES, log_toffoli_ladder_num_ancilla),
]
VARIANTS = {  # the methods of a ladder, with the options of its spec
    "ascending": {},
    "ascending_dagger": {"dagger": True},
    "descending": {"descending": True},
    "descending_dagger": {"descending": True, "dagger": True},
}
CASES += [
    (method(struct, variant), "n", partial(spec, **options), sizes, (), ancillas)
    for struct, spec, sizes, ancillas in LADDERS
    for variant, options in VARIANTS.items()
]
CASES += [  # the custom modifiers of the CX ladder
    (daggered_cx_ladder, "n", partial(cx_ladder, dagger=True), SIZES, (), 0, True),
    (controlled_cx_ladder, "qn", with_control(cx_ladder), SIZES, (), 0),
    (
        controlled_daggered_cx_ladder,
        "qn",
        partial(with_control(cx_ladder), dagger=True),
        SIZES,
        (),
        0,
    ),
]


# the machinery that turns each row of CASES into tests


BODIES = {  # what the wrapper runs in each mode
    None: "f({args})",
    "controlled": """\
with control(ctrl):
    f({args})""",
    "daggered": """\
f({args})
with dagger:
    f({args})""",
    "ctrl_daggered": """\
with control(ctrl):
    f({args})
    with dagger:
        f({args})""",
}


def wrap(f, n, shape, mode=None):
    """Wrap ``f`` at size ``n`` in a function that runs ``BODIES[mode]``."""
    shapes = {
        "q": "qubit",
        "n": f"array[qubit, {n}]",
        "i": f"array[qubit, {(n - 1).bit_length()}]",
    }
    types = [
        guppy.type_alias("Arg", s(n) if callable(s) else shapes.get(s, s).format(n=n))
        for s in shape
    ]
    names = "abcd"[: len(types)]
    body = textwrap.indent(BODIES[mode].format(args=", ".join(names)), "    ")
    params = ["ctrl: qubit"] * ("ctrl" in body) + [f"{x}: t{x}" for x in names]
    source = f"@guppy\ndef wrapped_{n}({', '.join(params)}) -> None:\n{body}\n"
    file = f"<wrapped {hash(source)}>"
    linecache.cache[file] = (len(source), None, source.splitlines(True), file)
    namespace = (
        globals() | {"f": f} | {f"t{x}": t for x, t in zip(names, types, strict=True)}
    )
    exec(compile(source, file, "exec"), namespace)
    return namespace[f"wrapped_{n}"]


def cases():
    """Yield each component with its wrappers by size."""
    for component, shape, spec, sizes, zero, ancillas, *modifiers in CASES:
        factory = inspect.isfunction(component)
        name = component.__name__ if factory else component.wrapped.name
        wrappers = {
            n: wrap(component(n) if factory else component, n, shape) for n in sizes
        }
        yield pytest.param(wrappers, spec, zero, ancillas, id=name)
        if any(modifiers):
            specs = {
                "controlled": with_control(spec),
                "daggered": same,
                "ctrl_daggered": same,
            }
            for mode, mode_spec in specs.items():
                wrappers = {
                    n: wrap(component(n) if factory else component, n, shape, mode)
                    for n in sorted({sizes[0], sizes[len(sizes) // 2], sizes[-1]})
                }
                yield pytest.param(wrappers, mode_spec, zero, None, id=f"{name}-{mode}")


@pytest.mark.parametrize(("wrappers", "spec", "zero", "ancillas"), list(cases()))
def test_component(wrappers, spec, zero, ancillas, subtests):
    """Compile the sizes of a component together, then check each against its spec."""
    gc.disable()  # it would walk the module again and again while it is built
    try:
        hugr = compile_together(wrappers.values())
    except Exception:
        hugr = None  # a size does not compile: each compiles alone, to tell which
    finally:
        gc.enable()
    for n, wrapper in wrappers.items():
        count = ancillas(n) if callable(ancillas) else ancillas
        with subtests.test(n=n) if len(wrappers) > 1 else nullcontext():
            try:
                verify(wrapper, spec, zero, hugr or compile_together([wrapper]), count)
            except Exception as error:
                raise error.with_traceback(None) from None


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize(
    ("k", "n"), [(1, 6), (2, 2), (3, 3), (4, 4), (5, 2), (6, 3), (7, 3), (8, 2), (9, 4)]
)
def test_cnx_ladder_logdepth(k, n, inverse):
    """Build the log depth ladder of k C^nX gates, then check it against cnx_ladder."""
    ladder = cnx_ladder_logdepth(k, n, inverse)
    m, n_ancillae = n - 1, cnx_ladder_logdepth_num_ancilla(k)

    @guppy
    @no_type_check
    def wrapped(
        controls_a: array[qubit, k],
        controls_b: array[array[qubit, m], k],
        target: qubit,
        borrowed_a: array[qubit, k],
        borrowed_b: array[qubit, k],
        ancillae: array[qubit, n_ancillae],
    ) -> None:
        ladder(controls_a, controls_b, target, borrowed_a, borrowed_b, ancillae)

    verify(wrapped, partial(cnx_ladder, inverse=inverse), ("ancillae",), ancillas=0)


@pytest.mark.parametrize(
    ("m", "n", "base"), [(2, 0, 3), (2, 1, 3), (3, 2, 3), (5, 3, 5)]
)
def test_exponentiator_shapes(m, n, base):
    """Check the exponentiator with m exponent qubits and n output qubits."""

    @guppy
    @no_type_check
    def wrapped(e: array[qubit, m], a: array[qubit, n]) -> None:
        exponentiator_ripple_gidney_mod(e, a, base)

    ancillas = 3 * n + 1 if n else 0
    verify(wrapped, partial(multiply_by_power, base=base), ancillas=ancillas)
