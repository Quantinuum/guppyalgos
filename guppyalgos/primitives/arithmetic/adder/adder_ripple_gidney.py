"""Ripple-carry Gidney quantum addition and subtraction circuits.

This module provides a Gidney ripple-carry [1 implementation, including modular variants
and non-modular circuits that expose a final carry/borrow qubit.

References:
    [1] Gidney, C. (2018). Halving the cost of quantum addition. Quantum, 2, 74.

"""

from typing import no_type_check

from guppylang import guppy
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.quantum import cx, discard_array, qubit, discard

from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.utils import qarray


@guppy
@no_type_check
def _cxx(control: qubit, target1: qubit, target2: qubit) -> None:
    cx(control, target1)
    cx(control, target2)


@guppy
@no_type_check
def _ccx(ctrl: qubit, x: qubit, y: qubit) -> None:
    """Toffoli gate via Gidney's measurement-based AND: y ^= ctrl AND x (4T gates).

    Uses ``temp_and_compute`` (4T, ancilla starts in ``|0⟩``) followed by
    ``cx`` and ``temp_and_uncompute`` (0T), giving a total cost of 4T gates.
    The ancilla qubit is allocated internally and consumed by the measurement
    in ``temp_and_uncompute``.
    """
    anc = qubit()
    temp_and_compute(ctrl, x, anc)
    cx(anc, y)
    temp_and_uncompute(ctrl, x, anc)
    discard(anc)


@guppy
@no_type_check
def _temp_and_cccx(ctrl: qubit, x: qubit, y: qubit, z: qubit) -> None:
    """Triply-controlled temporary AND gate via Gidney's measurement-based AND."""
    anc = qubit()
    temp_and_compute(ctrl, x, anc)
    temp_and_compute(anc, y, z)
    temp_and_uncompute(ctrl, x, anc)
    discard(anc)


@guppy
@no_type_check
def _temp_and_cccx_uncompute(ctrl: qubit, x: qubit, y: qubit, z: qubit) -> None:
    """Triply-controlled uncomputation of AND gate."""
    anc = qubit()
    temp_and_compute(ctrl, x, anc)
    temp_and_uncompute(anc, y, z)
    temp_and_uncompute(ctrl, x, anc)
    discard(anc)


@guppy
@no_type_check
def _cccx_toggle(
    ctrl: qubit,
    x: qubit,
    y: qubit,
    z: qubit,
) -> None:
    """Implement z ^= ctrl & x & y."""
    anc = qubit()
    temp_and_compute(ctrl, x, anc)
    _ccx(anc, y, z)
    temp_and_uncompute(ctrl, x, anc)
    discard(anc)


@guppy
@no_type_check
def _g_majority_gate(a: qubit, b: qubit, c: qubit, carry_out: qubit) -> None:
    """Gidney-majority gate.

    This gate propagates the majority of ``a``, ``b``, and ``c`` into the ripple.
    The majority outputs 1 if at least two of the three inputs are 1.

    Circuit:
        c: ──■────■───────■─────
             │    │       │
        a: ──X────┼───■───┼─────
                  │   │   │
        b: ───────X───■───┼─────
                      │   │
        0:             ───X─────

    """
    _cxx(c, a, b)
    temp_and_compute(a, b, carry_out)
    cx(c, carry_out)


@guppy
@no_type_check
def _g_majority_inverse_gate(a: qubit, b: qubit, c: qubit, carry_out: qubit) -> None:
    """Inverse of the Gidney-majority gate.

    This gate applies the inverse of _g_majority_gate.

    Circuit:
        c: ──■────────■────■─────
             │        │    │
        a: ──┼───■────X────┼─────
             │   │         │
        b: ──┼───■─────────X─────
             │   │
        C: ──X───

    """
    cx(c, carry_out)
    temp_and_uncompute(a, b, carry_out)
    _cxx(c, a, b)


@guppy
@no_type_check
def _g_unmajority_gate(a: qubit, b: qubit, c: qubit, carry_out: qubit) -> None:
    """G-unmajority gate.

    This gate reverses the operation of the G-majority gate while producing the sum bit.

    This does not include the final cx(a, b) gate required to finish computing the sum.

    Circuit:
        c: ──■────────■──────────
             │        │
        a: ──┼───■────X────■─────
             │   │         │
        b: ──┼───■─────────X─────
             │   │
        C: ──X───

    """
    cx(c, carry_out)
    temp_and_uncompute(a, b, carry_out)
    cx(c, a)
    cx(a, b)


@guppy
@no_type_check
def _g_unmajority_inverse_gate(a: qubit, b: qubit, c: qubit, carry_out: qubit) -> None:
    """Inverse of the Gidney-unmajority gate.

    This gate applies the inverse of _g_unmajority_gate.

    Circuit:
        c: ───────■───────■─────
                  │       │
        a: ──■────X───■───┼─────
             │        │   │
        b: ──X────────■───┼─────
                      │   │
        0:             ───X─────

    """
    cx(a, b)
    cx(c, a)
    temp_and_compute(a, b, carry_out)
    cx(c, carry_out)


@guppy
@no_type_check
def _gmaj_gumaj_bottom_carry_out_gate(
    a: qubit, b: qubit, c: qubit, carry_out: qubit
) -> None:
    """Apply the merged final GMAJ/controlled GUMAJ step for addition.

    Circuit:
        c: ──■────■───────■────■──────────
             │    │       │    │
        a: ──X────┼───■───┼────X────■─────
                  │   │   │         │
        b: ───────X───■───┼─────────X─────
                      │   │
        0:             ───X───────────────

    Args:
        a: Most-significant qubit of the first input register.
        b: Most-significant qubit of the second input/output register.
        c: Carry qubit from the preceding ripple stage.
        carry_out: Final carry-out.

    """
    _cxx(c, a, b)
    temp_and_compute(a, b, carry_out)
    cx(c, carry_out)
    cx(c, a)
    cx(a, b)


@guppy
@no_type_check
def _gmaj_gumaj_bottom_carry_out_inverse_gate(
    a: qubit,
    b: qubit,
    c: qubit,
    carry_out: qubit,
    uncompute_carry_out: bool,
) -> None:
    """Apply the inverse of _gmaj_gumaj_bottom_carry_out_gate.

    Circuit:
        c: ───────■───────■────■────■─────
                  │       │    │    │
        a: ──■────X───■───┼────X────┼─────
             │        │   │         │
        b: ──X────────■───┼─────────X─────
                      │   │
        0:             ───X───────────────

    Args:
        a: Most-significant qubit of the first input register.
        b: Most-significant qubit of the second input/output register.
        c: Carry qubit from the preceding ripple stage.
        carry_out: Final carry-out.
        uncompute_carry_out: If `True`, uncompute the carry-out already stored in
            `carry_out`. If `False`, `carry_out` must be a clean ancilla and is
            populated with the carry-out produced by the inverse circuit.

    """
    cx(a, b)
    cx(c, a)

    if uncompute_carry_out:
        cx(c, carry_out)
        temp_and_uncompute(a, b, carry_out)
    else:
        temp_and_compute(a, b, carry_out)
        cx(c, carry_out)

    _cxx(c, a, b)


@guppy
@no_type_check
def adder_ripple_gidney_mod[n: nat](
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
) -> None:
    """Apply a modular Gidney ripple-carry adder.

    This circuit performs in-place modular addition of a_reg into b_reg:

    ``|a>|b> -> |a>|a + b mod 2^n>``

    The circuit has linear depth and requires n - 1 ancilla qubits.

    Reference:
    Gidney, C. (2018). Halving the cost of quantum addition. Quantum, 2, 74.

    Args:
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg += a_reg mod 2^n.

    """
    if n == 1:
        cx(a_reg[0], b_reg[0])
        return

    anc = qarray(comptime(n - 1))

    temp_and_compute(a_reg[0], b_reg[0], anc[0])

    for i in range(n - 2):
        _g_majority_gate(a_reg[i + 1], b_reg[i + 1], anc[i], anc[i + 1])

    cx(a_reg[n - 1], b_reg[n - 1])
    cx(anc[n - 2], b_reg[n - 1])

    if n > 2:
        for i in range(n - 2):
            _g_unmajority_gate(
                a_reg[n - 2 - i], b_reg[n - 2 - i], anc[n - 3 - i], anc[n - 2 - i]
            )

    temp_and_uncompute(a_reg[0], b_reg[0], anc[0])
    cx(a_reg[0], b_reg[0])

    discard_array(anc)


@guppy
@no_type_check
def adder_ripple_gidney_mod_dagger[n: nat](
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
) -> None:
    """Apply the inverse of the modular Gidney ripple-carry adder.

    This circuit performs subtraction of ``a_reg`` from ``b_reg`` and
    computes the final borrow bit:

    ``|a>|b + a mod 2^n> -> |a>|b>``

    The circuit has linear depth and requires n - 1 ancilla qubits.

    Reference:
    Gidney, C. (2018). Halving the cost of quantum addition. Quantum, 2, 74.

    Args:
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg -= a_reg mod 2^n.

    """
    if n == 1:
        cx(a_reg[0], b_reg[0])
        return

    anc = qarray(comptime(n - 1))

    cx(a_reg[0], b_reg[0])
    temp_and_compute(a_reg[0], b_reg[0], anc[0])

    for i in range(n - 2):
        _g_unmajority_inverse_gate(a_reg[i + 1], b_reg[i + 1], anc[i], anc[i + 1])

    cx(a_reg[n - 1], b_reg[n - 1])
    cx(anc[n - 2], b_reg[n - 1])

    for i in range(n - 2):
        _g_majority_inverse_gate(
            a_reg[n - 2 - i], b_reg[n - 2 - i], anc[n - 3 - i], anc[n - 2 - i]
        )

    temp_and_uncompute(a_reg[0], b_reg[0], anc[0])

    discard_array(anc)


@guppy
@no_type_check
def adder_ripple_gidney_carry_out[n: nat](
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
    carry_out: qubit,
) -> None:
    """Apply a Gidney non-modular ripple-carry adder.

    The circuit implements the transformation:

    ``|a>|b>|0> -> |a>|b + a mod 2^n>|carry_out>``

    Args:
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg += a_reg mod 2^n.
        carry_out (qubit): Qubit that is flipped if the addition produces a final carry
            out.

    """
    anc = qarray(comptime(n - 1))

    if n == 1:
        temp_and_compute(a_reg[0], b_reg[0], carry_out)
        cx(a_reg[0], b_reg[0])
        discard_array(anc)
        return

    # ------------------------------------------------------------------ #
    # Forward sweep                                                      #
    # ------------------------------------------------------------------ #
    temp_and_compute(a_reg[0], b_reg[0], anc[0])

    for i in range(n - 2):
        _g_majority_gate(a_reg[i + 1], b_reg[i + 1], anc[i], anc[i + 1])

    _gmaj_gumaj_bottom_carry_out_gate(a_reg[n - 1], b_reg[n - 1], anc[n - 2], carry_out)

    # ------------------------------------------------------------------ #
    # Reverse sweep                                                      #
    # ------------------------------------------------------------------ #
    for i in range(n - 2):
        _g_unmajority_gate(
            a_reg[n - 2 - i], b_reg[n - 2 - i], anc[n - 3 - i], anc[n - 2 - i]
        )

    temp_and_uncompute(a_reg[0], b_reg[0], anc[0])
    cx(a_reg[0], b_reg[0])

    discard_array(anc)


@guppy
@no_type_check
def _adder_ripple_gidney_carry_out_dagger_impl[n: nat](
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
    carry_out: qubit,
    uncompute_carry_out: bool,
) -> None:
    """Apply the dagger of the non-modular Gidney ripple-carry adder.

    The circuit implements the transformation:

    ``|a>|b + a mod 2^n>|carry_out> -> |a>|b>|0>``

    Args:
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg += a_reg mod 2^n.
        carry_out (qubit): Qubit that is flipped iff the subtraction produces a
            final borrow (i.e. iff ``b < a``).
        uncompute_carry_out: If `True`, uncompute the carry-out already stored in
            `carry_out`. If `False`, `carry_out` must be a clean ancilla and is
            populated with the carry-out produced by the inverse circuit.

    """
    anc = qarray(comptime(n - 1))

    if n == 1:
        cx(a_reg[0], b_reg[0])
        if uncompute_carry_out:
            temp_and_uncompute(a_reg[0], b_reg[0], carry_out)
        else:
            temp_and_compute(a_reg[0], b_reg[0], carry_out)
        discard_array(anc)
        return

    cx(a_reg[0], b_reg[0])
    temp_and_compute(a_reg[0], b_reg[0], anc[0])

    # ------------------------------------------------------------------ #
    # Forward sweep                                                      #
    # ------------------------------------------------------------------ #
    for i in range(n - 2):
        _g_unmajority_inverse_gate(a_reg[i + 1], b_reg[i + 1], anc[i], anc[i + 1])

    _gmaj_gumaj_bottom_carry_out_inverse_gate(
        a_reg[n - 1], b_reg[n - 1], anc[n - 2], carry_out, uncompute_carry_out
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
def adder_ripple_gidney_carry_out_dagger[n: nat](
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
    carry_out: qubit,
) -> None:
    """Apply the dagger of the non-modular Gidney ripple-carry adder.

    The circuit implements the transformation:

    ``|a>|b + a mod 2^n>|carry_out> -> |a>|b>|0>``

    Args:
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg += a_reg mod 2^n.
        carry_out (qubit): Qubit that is flipped iff the subtraction produces a
            final borrow (i.e. iff ``b < a``).

    """
    return _adder_ripple_gidney_carry_out_dagger_impl(a_reg, b_reg, carry_out, True)


@guppy
@no_type_check
def _cntrl_g_unmajority_gate(
    ctrl: qubit, a: qubit, b: qubit, c: qubit, carry_out: qubit
) -> None:
    """Control G-unmajority gate."""
    cx(c, carry_out)
    temp_and_uncompute(a, b, carry_out)

    _ccx(ctrl, a, b)
    _cxx(c, a, b)


@guppy
@no_type_check
def _cntrl_g_majority_inverse_gate(
    ctrl: qubit, a: qubit, b: qubit, c: qubit, carry_out: qubit
) -> None:
    """Control the inverse G-majority gate."""
    cx(c, carry_out)
    temp_and_uncompute(a, b, carry_out)

    _ccx(ctrl, a, b)
    cx(c, a)
    cx(a, b)


@guppy
@no_type_check
def _cntrl_gmaj_gumaj_bottom_mod_gate(
    ctrl: qubit, a: qubit, b: qubit, c: qubit
) -> None:
    """Apply merged final GMAJ/controlled GUMAJ step for controlled modular addition.

    Args:
        ctrl: The control qubit.
        a: Most-significant qubit of the first input register.
        b: Most-significant qubit of the second input/output register.
        c: Carry qubit from the preceding ripple stage.

    """
    cx(c, a)
    _ccx(ctrl, a, b)
    cx(c, a)


@guppy
@no_type_check
def _cntrl_gmaj_gumaj_bottom_carry_out_gate(
    ctrl: qubit, a: qubit, b: qubit, c: qubit, carry_out: qubit
) -> None:
    """Apply the merged final GMAJ/controlled GUMAJ step for controlled addition.

    Args:
        ctrl: The control qubit.
        a: Most-significant qubit of the first input register.
        b: Most-significant qubit of the second input/output register.
        c: Carry qubit from the preceding ripple stage.
        carry_out: Final carry-out.

    """
    _cxx(c, a, b)

    _temp_and_cccx(ctrl, a, b, carry_out)
    _ccx(ctrl, c, carry_out)
    _ccx(ctrl, a, b)

    _cxx(c, a, b)


@guppy
@no_type_check
def _cntrl_gmaj_gumaj_bottom_carry_out_inverse_gate(
    ctrl: qubit,
    a: qubit,
    b: qubit,
    c: qubit,
    carry_out: qubit,
    uncompute_carry_out: bool,
) -> None:
    """Apply the inverse of _cntrl_gmaj_gumaj_bottom_carry_out_gate.

    Args:
        ctrl: The control qubit.
        a: Most-significant qubit of the first input register.
        b: Most-significant qubit of the second input/output register.
        c: Carry qubit from the preceding ripple stage.
        carry_out: Final carry-out.
        uncompute_carry_out: If `True`, uncompute the carry-out already stored in
            `carry_out`. If `False`, `carry_out` must be a clean ancilla and is
            populated with the carry-out produced by the inverse circuit.

    """
    cx(a, b)
    cx(c, a)

    if uncompute_carry_out:
        _ccx(ctrl, c, carry_out)
        _temp_and_cccx_uncompute(ctrl, a, b, carry_out)
    else:
        _temp_and_cccx(ctrl, a, b, carry_out)
        _ccx(ctrl, c, carry_out)

    _ccx(ctrl, a, b)

    cx(c, a)
    cx(a, b)


@guppy
@no_type_check
def cntrl_adder_ripple_gidney_mod[n: nat](
    ctrl: qubit,
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
) -> None:
    """Apply a controlled modular Gidney ripple-carry adder.

    The circuit implements the transformation:

    ``|ctrl>|a>|b> -> |ctrl>|a>|b + ctrl * a mod 2^n>``

    Implements the controlled modular addition circuit from Figure 4 of
    https://quantum-journal.org/papers/q-2018-06-18-74/pdf/.

    Each intermediate bit gadget costs 8T gates: one ``temp_and_compute`` for
    the carry and one temporary logical-AND pair in the backward pass for the
    controlled addend contribution.

    Args:
        ctrl (qubit): The control qubit.
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg += a_reg mod 2^n.

    """
    if n == 1:
        _ccx(ctrl, a_reg[0], b_reg[0])
        return

    anc = qarray(comptime(n - 1))

    # ------------------------------------------------------------------ #
    # Forward sweep                                                      #
    # ------------------------------------------------------------------ #
    temp_and_compute(a_reg[0], b_reg[0], anc[0])

    for i in range(n - 2):
        _g_majority_gate(a_reg[i + 1], b_reg[i + 1], anc[i], anc[i + 1])

    _cntrl_gmaj_gumaj_bottom_mod_gate(ctrl, a_reg[n - 1], b_reg[n - 1], anc[n - 2])

    # ------------------------------------------------------------------ #
    # Reverse sweep                                                      #
    # ------------------------------------------------------------------ #
    for i in range(n - 2):
        _cntrl_g_unmajority_gate(
            ctrl, a_reg[n - 2 - i], b_reg[n - 2 - i], anc[n - 3 - i], anc[n - 2 - i]
        )

    temp_and_uncompute(a_reg[0], b_reg[0], anc[0])
    _ccx(ctrl, a_reg[0], b_reg[0])

    discard_array(anc)


@guppy
@no_type_check
def cntrl_adder_ripple_gidney_mod_dagger[n: nat](
    ctrl: qubit,
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
) -> None:
    """Apply the dagger of the controlled modular Gidney ripple-carry adder.

    The circuit implements the transformation:

    ``|ctrl>|a>|b + ctrl * a mod 2^n> -> |ctrl>|a>|b>``

    Implements the inverse of the controlled modular addition circuit from Figure 4 of
    https://quantum-journal.org/papers/q-2018-06-18-74/pdf/.

    Args:
        ctrl (qubit): The control qubit.
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg += a_reg mod 2^n.

    """
    if n == 1:
        _ccx(ctrl, a_reg[0], b_reg[0])
        return

    anc = qarray(comptime(n - 1))

    # ------------------------------------------------------------------ #
    # Forward sweep                                                      #
    # ------------------------------------------------------------------ #
    _ccx(ctrl, a_reg[0], b_reg[0])
    temp_and_compute(a_reg[0], b_reg[0], anc[0])

    for i in range(n - 2):
        _g_unmajority_inverse_gate(a_reg[i + 1], b_reg[i + 1], anc[i], anc[i + 1])

    _cntrl_gmaj_gumaj_bottom_mod_gate(ctrl, a_reg[n - 1], b_reg[n - 1], anc[n - 2])

    # ------------------------------------------------------------------ #
    # Reverse sweep                                                      #
    # ------------------------------------------------------------------ #
    for i in range(n - 2):
        _cntrl_g_majority_inverse_gate(
            ctrl, a_reg[n - 2 - i], b_reg[n - 2 - i], anc[n - 3 - i], anc[n - 2 - i]
        )

    temp_and_uncompute(a_reg[0], b_reg[0], anc[0])

    discard_array(anc)


@guppy
@no_type_check
def cntrl_adder_ripple_gidney_carry_out[n: nat](
    ctrl: qubit,
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
    carry_out: qubit,
) -> None:
    """Apply a controlled Gidney ripple-carry adder.

    The circuit implements the transformation:

    ``|ctrl>|a>|b>|0> -> |ctrl>|a>|b + ctrl * a mod 2^n>|ctrl * carry_out>``

    Args:
        ctrl (qubit): The control qubit.
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg += a_reg mod 2^n.
        carry_out (qubit): Qubit that is flipped if the addition produces a final carry
            out.

    """
    anc = qarray(comptime(n - 1))

    # ------------------------------------------------------------------ #
    # Forward sweep                                                      #
    # ------------------------------------------------------------------ #
    temp_and_compute(a_reg[0], b_reg[0], anc[0])

    for i in range(n - 2):
        _g_majority_gate(a_reg[i + 1], b_reg[i + 1], anc[i], anc[i + 1])

    _cntrl_gmaj_gumaj_bottom_carry_out_gate(
        ctrl, a_reg[n - 1], b_reg[n - 1], anc[n - 2], carry_out
    )

    # ------------------------------------------------------------------ #
    # Reverse sweep                                                      #
    # ------------------------------------------------------------------ #
    for i in range(n - 2):
        _cntrl_g_unmajority_gate(
            ctrl, a_reg[n - 2 - i], b_reg[n - 2 - i], anc[n - 3 - i], anc[n - 2 - i]
        )

    temp_and_uncompute(a_reg[0], b_reg[0], anc[0])
    _ccx(ctrl, a_reg[0], b_reg[0])

    discard_array(anc)


@guppy
@no_type_check
def _cntrl_adder_ripple_gidney_carry_out_dagger_impl[n: nat](
    ctrl: qubit,
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
    carry_out: qubit,
    uncompute_carry_out: bool,
) -> None:
    """Apply the dagger of the controlled Gidney ripple-carry adder.

    The circuit implements the transformation:

    ``|ctrl>|a>|b + ctrl * a mod 2^n>|ctrl * carry_out> -> |ctrl>|a>|b>|0>``

    Args:
        ctrl (qubit): The control qubit.
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg += a_reg mod 2^n.
        carry_out (qubit): Qubit that is flipped iff the subtraction produces a
            final borrow (i.e. iff ``b < a``).
        uncompute_carry_out: If `True`, uncompute the carry-out already stored in
            `carry_out`. If `False`, `carry_out` must be a clean ancilla and is
            populated with the carry-out produced by the inverse circuit.

    """
    anc = qarray(comptime(n - 1))

    _ccx(ctrl, a_reg[0], b_reg[0])
    temp_and_compute(a_reg[0], b_reg[0], anc[0])

    # ------------------------------------------------------------------ #
    # Forward sweep                                                      #
    # ------------------------------------------------------------------ #
    for i in range(n - 2):
        _g_unmajority_inverse_gate(a_reg[i + 1], b_reg[i + 1], anc[i], anc[i + 1])

    _cntrl_gmaj_gumaj_bottom_carry_out_inverse_gate(
        ctrl, a_reg[n - 1], b_reg[n - 1], anc[n - 2], carry_out, uncompute_carry_out
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


@guppy
@no_type_check
def cntrl_adder_ripple_gidney_carry_out_dagger[n: nat](
    ctrl: qubit,
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
    carry_out: qubit,
) -> None:
    """Apply the dagger of the controlled Gidney ripple-carry adder.

    The circuit implements the transformation:

    ``|ctrl>|a>|b + ctrl * a mod 2^n>|ctrl * carry_out> -> |ctrl>|a>|b>|0>``

    Args:
        ctrl (qubit): The control qubit.
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg += a_reg mod 2^n.
        carry_out (qubit): Qubit that is flipped iff the subtraction produces a
            final borrow (i.e. iff ``b < a``).

    """
    return _cntrl_adder_ripple_gidney_carry_out_dagger_impl(
        ctrl, a_reg, b_reg, carry_out, True
    )
