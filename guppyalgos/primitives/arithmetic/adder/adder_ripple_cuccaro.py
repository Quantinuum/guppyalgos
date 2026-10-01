"""Ripple-carry quantum addition and subtraction circuits.

This module provides majority/unmajority [1] and optimized Cuccaro ripple-carry [2]
implementations, including modular variants and non-modular circuits that expose a final
carry/borrow qubit. Small registers use the majority/unmajority construction
as a fallback.

References:
    [1] Vedral, V., Barenco, A., & Ekert, A. (1996). Quantum networks for
        elementary arithmetic operations. Physical Review A, 54(1), 147.
    [2] Cuccaro, S. A., et al. (2004). A new quantum ripple-carry addition
        circuit. arXiv:quant-ph/0410184.

"""

from typing import no_type_check

from guppylang import guppy
from guppyalgos.utils import apply_bitstring, transversal
from guppylang.std.quantum import qubit, cx, x, toffoli, discard
from guppylang.std.builtins import array, frozenarray, nat
from guppyalgos.utils import qarray
from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_compute,
    temp_and_uncompute,
)
from guppyalgos.primitives.subroutines.ladders import CXLadderLinear
from guppyalgos.utils.guppy.unsafe_borrow import (
    _unsafe_array_borrow_slice,
    _unsafe_array_unborrow_slice,
)


@guppy
@no_type_check
def _crc_prep_regs[n: nat](
    a_bit_array: frozenarray[bool, n], b_bit_array: frozenarray[bool, n]
) -> tuple[array[qubit, n], array[qubit, n], qubit]:
    """Prepare the registers for ripple carry adder."""
    a_reg = qarray(n)
    apply_bitstring(a_reg, a_bit_array)
    b_reg = qarray(n)
    apply_bitstring(b_reg, b_bit_array)
    carry_out = qubit()

    return a_reg, b_reg, carry_out


@guppy(unitary=True)
@no_type_check
def _majority_gate(a: qubit, b: qubit, c: qubit) -> None:
    """Majority gate.

    This gate propagates the majority of ``a``, ``b``, and ``c`` into the ripple.
    The majority outputs 1 if at least two of the three inputs are 1.

    Circuit:
        a: ──■────■───X─────────
             │    │   │
        b: ──X────┼───■─────────
                  │   │
        c: ───────X───■─────────

    """
    cx(a, b)
    cx(a, c)
    toffoli(c, b, a)


@guppy(unitary=True)
@no_type_check
def _unmajority_gate(a: qubit, b: qubit, c: qubit) -> None:
    """Unmajority gate.

    This gate reverses the operation of the majority gate while producing the sum bit.

    Circuit:
        a: ──────X────■─────────
                 │    │
        b: ──────■────┼───X─────
                 │    │   │
        c: ──────■────X───■─────

    """
    toffoli(c, b, a)
    cx(a, c)
    cx(c, b)


@guppy
@no_type_check
def _maj_umaj_bottom_carry_out_gate(
    a: qubit, b: qubit, c: qubit, carry_out: qubit
) -> None:
    """Apply the merged final MAJ/UMAJ step of a ripple-carry adder.

    This gate acts on the most-significant bit of the addition. It is
    equivalent to applying a majority gate, copying the resulting final
    carry into ``carry_out``, and then applying the corresponding
    unmajority gate.

    The common operations in the MAJ and UMAJ gates are canceled, giving
    the reduced sequence implemented below. The input qubits ``a``
    and ``c`` are restored as required by the surrounding ripple, while
    ``carry_out`` is flipped iff the addition produces a final carry.

    Args:
        a: Most-significant qubit of the first input register.
        b: Most-significant qubit of the second input/output register.
        c: Carry qubit from the preceding ripple stage.
        carry_out: Output qubit receiving the final carry bit.

    Circuit:
        a: ──■────■───────■────■──────────
             │    │       │    │
        b: ──X────┼───■───┼────┼────X─────
                  │   │   │    │    │
        c: ───────X───■───┼────X────■─────
                      │   │
        0: ───────────X───X───────────────

    """
    cx(a, b)
    cx(a, c)

    temp_and_compute(c, b, carry_out)
    cx(a, carry_out)

    cx(a, c)
    cx(c, b)


@guppy
@no_type_check
def _maj_umaj_bottom_carry_out_inverse_gate(
    a: qubit, b: qubit, c: qubit, carry_out: qubit, uncompute_carry_out: bool
) -> None:
    """Apply the merged final MAJ/UMAJ step of a ripple-carry subtractor.

    This gate acts on the most-significant bit of the subtraction. It is
    equivalent to applying a majority† gate, copying the resulting final
    carry into ``carry_out``, and then applying the corresponding
    unmajority gate.

    The common operations in the MAJ and UMAJ gates are canceled, giving
    the reduced sequence implemented below. The input qubits ``a``
    and ``c`` are restored as required by the surrounding ripple, while
    ``carry_out`` is flipped iff the addition produces a final carry.

    Args:
        a: Most-significant qubit of the first input register.
        b: Most-significant qubit of the second input/output register.
        c: Carry qubit from the preceding ripple stage.
        carry_out: Output qubit receiving the final carry bit.
        uncompute_carry_out: If `True`, uncompute the carry-out already stored in
            `carry_out`. If `False`, `carry_out` must be a clean ancilla and is
            populated with the carry-out produced by the inverse circuit.


    Circuit:
        a: ───────■───────■────■────■────
                  │       │    │    │
        b: ──X────┼───■───┼────┼────X─────
             │    │   │   │    │    │
        c: ──■────X───■───┼────X────■─────
                      │   │
        0: ───────────X───X───────────────

    """
    cx(c, b)
    cx(a, c)

    if uncompute_carry_out:
        cx(a, carry_out)
        temp_and_uncompute(c, b, carry_out)
    else:
        temp_and_compute(c, b, carry_out)
        cx(a, carry_out)

    cx(a, c)
    cx(a, b)


@guppy(unitary=True)
@no_type_check
def _maj_umaj_bottom_mod_gate(a: qubit, b: qubit, c: qubit) -> None:
    """Apply the merged final MAJ/UMAJ step for modular addition.

    This gate acts on the most-significant bit of a ripple-carry adder when
    the final carry is discarded, i.e. for addition modulo ``2**n``.

    It is equivalent to applying the final majority gate followed immediately
    by its corresponding unmajority gate, with no carry-out extraction between
    them. Canceling the redundant operations leaves the two CNOT gates below.

    Args:
        a: Most-significant qubit of the first input register.
        b: Most-significant qubit of the second input/output register.
        c: Carry qubit from the preceding ripple stage.

    Circuit:
        a: ──■──────────
             │
        b: ──X────X─────
                  │
        c: ───────■─────

    """
    cx(a, b)
    cx(c, b)


@guppy(unitary=True)
@no_type_check
def _cntrl_majority_dagger_gate(ctrl: qubit, a: qubit, b: qubit, c: qubit) -> None:
    """Control majority† gate.

    Circuit:
     ctrl: ───────────■───────────────
                      │
        a: ──────X────┼────■──────────
                 │    │    │
        b: ──────■────X────┼────X─────
                 │    │    │    │
        c: ──────■────■────X────■─────

    """
    toffoli(c, b, a)
    toffoli(ctrl, c, b)
    cx(a, c)
    cx(c, b)


@guppy(unitary=True)
@no_type_check
def _cntrl_unmajority_gate(ctrl: qubit, a: qubit, b: qubit, c: qubit) -> None:
    """Control unmajority gate.

    This gate reverses the operation of the majority gate while conditionally producing
    the sum bit.

    Circuit:
     ctrl: ───────────■───────────────
                      │
        a: ──────X────┼────■────■─────
                 │    │    │    │
        b: ──────■────X────X────┼─────
                 │    │         │
        c: ──────■────■─────────X─────

    """
    toffoli(c, b, a)
    toffoli(ctrl, c, b)
    cx(a, b)
    cx(a, c)


@guppy(unitary=True)
@no_type_check
def _cntrl_maj_umaj_bottom_mod_gate(ctrl: qubit, a: qubit, b: qubit, c: qubit) -> None:
    """Apply the merged final MAJ/controlled UMAJ step for controlled modular addition.

    This gate acts on the most-significant bit of a ripple-carry adder when
    the final carry is discarded, i.e. for addition modulo ``2**n``.

    It is equivalent to applying the final majority gate followed immediately
    by its corresponding unmajority gate, with no carry-out extraction between
    them. Canceling the redundant operations leaves the two CNOT gates below.

    Args:
        ctrl: The control qubit.
        a: Most-significant qubit of the first input register.
        b: Most-significant qubit of the second input/output register.
        c: Carry qubit from the preceding ripple stage.

    Circuit:
     ctrl: ───────■──────────
                  │
        a: ──X────■────X─────
             │    │    │
        b: ──┼────X────┼─────
             │         │
        c: ──■─────────■─────

    """
    cx(c, a)
    toffoli(ctrl, a, b)
    cx(c, a)


@guppy
@no_type_check
def adder_ripple_cuccaro_mod[n: nat](
    a_reg: array[qubit, n], b_reg: array[qubit, n]
) -> None:
    """Apply an optimized modular Cuccaro ripple-carry adder.

    This circuit performs in-place modular addition of a_reg into b_reg:

    ``|a>|b> -> |a>|a + b mod 2^n>``

    The circuit has linear depth and requires one ancilla qubit.

    This is the modular case of adder_ripple_cuccaro_carry_out.

    Reference:
    Cuccaro, Steven A., et al. "A new quantum ripple-carry addition
    circuit." arXiv preprint quant-ph/0410184 (2004).

    Args:
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg += a_reg mod 2^n.

    """
    cx_ladder = CXLadderLinear()
    ancilla = qubit()

    if n == 1:
        cx(a_reg[0], b_reg[0])
        discard(ancilla)
        return

    if n == 2:
        # Boundary MAJ: carry-in is known to be 0 and ancilla is clean.
        temp_and_compute(a_reg[0], b_reg[0], ancilla)

        _maj_umaj_bottom_mod_gate(a_reg[1], b_reg[1], ancilla)

        # Boundary UMAJ: uncompute the clean ancilla.
        temp_and_uncompute(a_reg[0], b_reg[0], ancilla)
        cx(a_reg[0], b_reg[0])
        discard(ancilla)
        return

    # Boundary MAJ: carry-in is known to be 0 and ancilla is clean.
    temp_and_compute(a_reg[0], b_reg[0], ancilla)

    transversal(cx, a_reg, b_reg, 1, n)

    # Forward n-2 CX ladder
    cx(a_reg[1], ancilla)
    cx_p, cx_q, cx_s = _unsafe_array_borrow_slice(
        a_reg, comptime(min(1, n - 1)), comptime(max(0, n - 2)), 1
    )
    cx_ladder.descending_dagger(cx_q)
    _unsafe_array_unborrow_slice(a_reg, cx_p, cx_q, cx_s)

    # Forward n-2 Toffoli ladder
    toffoli(ancilla, b_reg[1], a_reg[1])
    for i in range(n - 3):
        toffoli(a_reg[i + 1], b_reg[i + 2], a_reg[i + 2])

    for i in range(n - 2):
        x(b_reg[i + 1])

    cx(ancilla, b_reg[1])
    for i in range(n - 2):
        cx(a_reg[i + 1], b_reg[i + 2])

    # Reverse n-2 Toffoli ladder
    for i in range(n - 3):
        toffoli(a_reg[n - i - 3], b_reg[n - i - 2], a_reg[n - i - 2])
    toffoli(ancilla, b_reg[1], a_reg[1])

    for i in range(n - 2):
        x(b_reg[i + 1])

    # Reverse n-2 CX ladder
    cx_p, cx_q, cx_s = _unsafe_array_borrow_slice(
        a_reg, comptime(min(1, n - 1)), comptime(max(0, n - 2)), 1
    )
    cx_ladder.descending(cx_q)
    _unsafe_array_unborrow_slice(a_reg, cx_p, cx_q, cx_s)
    cx(a_reg[1], ancilla)

    temp_and_uncompute(a_reg[0], b_reg[0], ancilla)

    transversal(cx, a_reg, b_reg, 0, comptime(n - 1))

    discard(ancilla)


@guppy
@no_type_check
def adder_ripple_cuccaro_mod_dagger[n: nat](
    a_reg: array[qubit, n], b_reg: array[qubit, n]
) -> None:
    """Apply the inverse of the Cuccaro ripple-carry modular adder circuit.

    This circuit is the inverse of the Cuccaro ripple-carry
    adder. It computes b - a in place on b_reg:

    ``|a>|b + a mod 2^n> -> |a>|b>``

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

    """
    cx_ladder = CXLadderLinear()
    ancilla = qubit()

    if n == 1:
        cx(a_reg[0], b_reg[0])
        discard(ancilla)
        return

    if n == 2:
        # Boundary UMAJ†: carry-in is known to be 0 and ancilla is clean.
        cx(a_reg[0], b_reg[0])
        temp_and_compute(a_reg[0], b_reg[0], ancilla)
        with dagger:
            _maj_umaj_bottom_mod_gate(a_reg[1], b_reg[1], ancilla)

        # Boundary MAJ†: uncompute the clean ancilla.
        temp_and_uncompute(a_reg[0], b_reg[0], ancilla)
        discard(ancilla)
        return

    transversal(cx, a_reg, b_reg, 0, comptime(n - 1))

    temp_and_compute(a_reg[0], b_reg[0], ancilla)

    # Forward n-2 CX ladder
    cx(a_reg[1], ancilla)
    cx_p, cx_q, cx_s = _unsafe_array_borrow_slice(
        a_reg, comptime(min(1, n - 1)), comptime(max(0, n - 2)), 1
    )
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

    # Reverse n-2 Toffoli ladder
    for i in range(n - 4, -1, -1):
        toffoli(
            a_reg[i + 1],
            b_reg[i + 2],
            a_reg[i + 2],
        )
    toffoli(ancilla, b_reg[1], a_reg[1])

    # Reverse n-2 CX ladder
    cx_p, cx_q, cx_s = _unsafe_array_borrow_slice(
        a_reg, comptime(min(1, n - 1)), comptime(max(0, n - 2)), 1
    )
    cx_ladder.descending(cx_q)
    _unsafe_array_unborrow_slice(a_reg, cx_p, cx_q, cx_s)
    cx(a_reg[1], ancilla)

    temp_and_uncompute(a_reg[0], b_reg[0], ancilla)

    transversal(cx, a_reg, b_reg, 1, n)

    discard(ancilla)


@guppy
@no_type_check
def adder_ripple_cuccaro_carry_out[n: nat](
    a_reg: array[qubit, n], b_reg: array[qubit, n], carry_out: qubit
) -> None:
    """Apply an optimized Cuccaro ripple-carry adder.

    This circuit performs in-place modular addition of a_reg into
    b_reg and computes the final carry-out:

    ``|a>|b>|0> -> |a>|a + b mod 2^n>|carry_out>``

    The circuit requires one ancilla qubit. The circuit's depth depends on the supplied
    ladder implementations.

    When both the CNOT and Toffoli ladders are linear, gates from the two ladders can
    be interleaved and executed in parallel, as shown in Fig. 6 of the Reference.
    The pseudocode in Fig. 5 makes this scheduling explicit: each line corresponds to a
    single time-slice.

    In our implementation, the ladders are represented separately rather than being
    manually interleaved. This provides greater flexibility: when linear ladders are
    supplied, the compiler can identify independent operations and schedule them in
    parallel, recovering the linear-depth construction described by Cuccaro et al. If
    alternative ladder implementations are supplied—for example, logarithmic-depth
    ladders—the surrounding adder construction does not impose the linear scheduling of
    Fig. 5 and can instead take advantage of the lower-depth ladder structure.

    Reference:
    Cuccaro, Steven A., et al. "A new quantum ripple-carry addition
    circuit." arXiv preprint quant-ph/0410184 (2004).

    Args:
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg += a_reg mod 2^n.
        carry_out (qubit): Qubit that is flipped if the addition produces a final carry
            out.

    """
    cx_ladder = CXLadderLinear()
    ancilla = qubit()

    if n == 1:
        temp_and_compute(a_reg[0], b_reg[0], carry_out)
        cx(a_reg[0], b_reg[0])
        discard(ancilla)
        return

    if n == 2:
        # Boundary MAJ: carry-in is known to be 0 and ancilla is clean.
        temp_and_compute(a_reg[0], b_reg[0], ancilla)

        _maj_umaj_bottom_carry_out_gate(a_reg[1], b_reg[1], ancilla, carry_out)

        # Boundary UMAJ: uncompute the clean ancilla.
        temp_and_uncompute(a_reg[0], b_reg[0], ancilla)
        cx(a_reg[0], b_reg[0])
        discard(ancilla)
        return

    temp_and_compute(a_reg[0], b_reg[0], ancilla)

    transversal(cx, a_reg, b_reg, 1, n)

    # Forward n-1 CX ladder
    cx(a_reg[1], ancilla)
    cx_p, cx_q, cx_s = _unsafe_array_borrow_slice(a_reg, 1, comptime(n - 1), 0)
    cx_ladder.descending_dagger(cx_q)
    _unsafe_array_unborrow_slice(a_reg, cx_p, cx_q, cx_s)

    # Forward n-2 Toffoli ladder
    toffoli(ancilla, b_reg[1], a_reg[1])
    if n > 2:
        for i in range(n - 3):
            toffoli(a_reg[i + 1], b_reg[i + 2], a_reg[i + 2])

    temp_and_compute(a_reg[n - 2], b_reg[n - 1], carry_out)
    cx(a_reg[n - 1], carry_out)

    for i in range(n - 2):
        x(b_reg[i + 1])

    cx(ancilla, b_reg[1])
    for i in range(n - 2):
        cx(a_reg[i + 1], b_reg[i + 2])

    # Reverse n-2 Toffoli ladder
    for i in range(n - 3):
        toffoli(a_reg[n - i - 3], b_reg[n - i - 2], a_reg[n - i - 2])
    toffoli(ancilla, b_reg[1], a_reg[1])

    for i in range(n - 2):
        x(b_reg[i + 1])

    # Reverse n-1 CX ladder
    cx_p, cx_q, cx_s = _unsafe_array_borrow_slice(a_reg, 1, comptime(n - 1), 0)
    cx_ladder.descending(cx_q)
    _unsafe_array_unborrow_slice(a_reg, cx_p, cx_q, cx_s)
    cx(a_reg[1], ancilla)

    temp_and_uncompute(a_reg[0], b_reg[0], ancilla)

    transversal(cx, a_reg, b_reg)

    discard(ancilla)


@guppy
@no_type_check
def _adder_ripple_cuccaro_carry_out_dagger_impl[n: nat](
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
    carry_out: qubit,
    uncompute_carry_out: bool,
) -> None:
    """Shared implementation for the Cuccaro carry-out subtraction variants.

    For ``n >= 4`` the two public variants use the same optimized circuit. For
    smaller registers, ``uncompute_borrow_out`` selects the corresponding
    majority/unmajority fallback behavior.

    Args:
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg -= a_reg mod 2^n.
        carry_out (qubit): Qubit that is flipped iff the subtraction produces a
            final borrow (i.e. iff ``b < a``).
        uncompute_carry_out: If `True`, uncompute the carry-out already stored in
            `carry_out`. If `False`, `carry_out` must be a clean ancilla and is
            populated with the carry-out produced by the inverse circuit.

    """
    cx_ladder = CXLadderLinear()
    ancilla = qubit()

    if n == 1:
        cx(a_reg[0], b_reg[0])
        if uncompute_carry_out:
            temp_and_uncompute(a_reg[0], b_reg[0], carry_out)
        else:
            temp_and_compute(a_reg[0], b_reg[0], carry_out)
        discard(ancilla)
        return

    if n == 2:
        # Boundary UMAJ†: carry-in is known to be 0 and ancilla is clean.
        cx(a_reg[0], b_reg[0])
        temp_and_compute(a_reg[0], b_reg[0], ancilla)

        _maj_umaj_bottom_carry_out_inverse_gate(
            a_reg[1], b_reg[1], ancilla, carry_out, uncompute_carry_out
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

    if uncompute_carry_out:
        cx(a_reg[n - 1], carry_out)
        temp_and_uncompute(a_reg[n - 2], b_reg[n - 1], carry_out)
    else:
        temp_and_compute(a_reg[n - 2], b_reg[n - 1], carry_out)
        cx(a_reg[n - 1], carry_out)

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
def adder_ripple_cuccaro_carry_out_dagger[n: nat](
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
    carry_out: qubit,
) -> None:
    """Apply the inverse of the Cuccaro ripple-carry adder.

    Computes ``b - a`` in place on ``b_reg`` while restoring ``a_reg``. The
    supplied ``borrow_out`` qubit is treated as the carry-out qubit from the
    forward adder and is uncomputed by the inverse operation.

    Transformation::

        |a>|a + b mod 2^n>|carry_out> -> |a>|b>|0>

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
        carry_out (qubit): Carry/borrow qubit to uncompute.

    """
    return _adder_ripple_cuccaro_carry_out_dagger_impl(a_reg, b_reg, carry_out, True)


@guppy
@no_type_check
def cntrl_adder_ripple_cuccaro_mod[n: nat](
    ctrl: qubit,
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
) -> None:
    """Apply a controlled optimized modular Cuccaro ripple-carry adder.

    The circuit implements the transformation:

    ``|ctrl>|a>|b> -> |ctrl>|a>|b + ctrl * a mod 2^n>``

    The circuit requires one ancilla qubit. The circuit's depth depends on the supplied
    ladder implementations.

    This is the modular variant of cntrl_adder_ripple_cuccaro_carry_out.

    Reference:
    Cuccaro, Steven A., et al. "A new quantum ripple-carry addition
    circuit." arXiv preprint quant-ph/0410184 (2004).

    Args:
        ctrl (qubit): The control qubit.
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg += a_reg mod 2^n.

    """
    cx_ladder = CXLadderLinear()
    ancilla = qubit()

    if n == 1:
        toffoli(ctrl, a_reg[0], b_reg[0])
        discard(ancilla)
        return

    if n == 2:
        # Boundary MAJ: carry-in is known to be 0 and ancilla is clean.
        temp_and_compute(a_reg[0], b_reg[0], ancilla)
        _cntrl_maj_umaj_bottom_mod_gate(ctrl, a_reg[1], b_reg[1], ancilla)

        # Boundary controlled-UMAJ: uncompute the clean ancilla.
        temp_and_uncompute(a_reg[0], b_reg[0], ancilla)
        toffoli(ctrl, a_reg[0], b_reg[0])
        discard(ancilla)
        return

    transversal(cx, a_reg, b_reg, 1, comptime(n - 1))

    temp_and_compute(a_reg[0], b_reg[0], ancilla)

    # Forward n-2 CX ladder
    cx(a_reg[1], ancilla)
    cx_p, cx_q, cx_s = _unsafe_array_borrow_slice(
        a_reg, comptime(min(1, n - 1)), comptime(max(0, n - 2)), 1
    )
    cx_ladder.descending_dagger(cx_q)
    _unsafe_array_unborrow_slice(a_reg, cx_p, cx_q, cx_s)

    # Forward n-2 Toffoli ladder
    toffoli(ancilla, b_reg[1], a_reg[1])
    for i in range(n - 3):
        toffoli(a_reg[i + 1], b_reg[i + 2], a_reg[i + 2])

    cx(a_reg[n - 2], a_reg[n - 1])
    toffoli(ctrl, a_reg[n - 1], b_reg[n - 1])
    cx(a_reg[n - 2], a_reg[n - 1])

    # Reverse n-2 Toffoli ladders
    for i in range(n - 3):
        toffoli(a_reg[n - i - 3], b_reg[n - i - 2], a_reg[n - i - 2])
        toffoli(ctrl, a_reg[n - i - 3], b_reg[n - i - 2])
    toffoli(ancilla, b_reg[1], a_reg[1])
    toffoli(ctrl, ancilla, b_reg[1])

    # Reverse n-2 CX ladder
    cx_p, cx_q, cx_s = _unsafe_array_borrow_slice(
        a_reg, comptime(min(1, n - 1)), comptime(max(0, n - 2)), 1
    )
    cx_ladder.descending(cx_q)
    _unsafe_array_unborrow_slice(a_reg, cx_p, cx_q, cx_s)
    cx(a_reg[1], ancilla)

    temp_and_uncompute(a_reg[0], b_reg[0], ancilla)

    transversal(cx, a_reg, b_reg, 1, comptime(n - 1))
    toffoli(ctrl, a_reg[0], b_reg[0])

    discard(ancilla)


@guppy
@no_type_check
def cntrl_adder_ripple_cuccaro_mod_dagger[n: nat](
    ctrl: qubit,
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
) -> None:
    """Apply the inverse of the controlled optimized modular Cuccaro ripple-carry adder.

    The circuit implements the transformation:

    ``|ctrl>|a>|b + ctrl * a mod 2^n> -> |ctrl>|a>|b>``

    The circuit requires one ancilla qubit. The circuit's depth depends on the supplied
    ladder implementations.

    This is the modular variant of cntrl_adder_ripple_cuccaro_carry_out.

    Reference:
    Cuccaro, Steven A., et al. "A new quantum ripple-carry addition
    circuit." arXiv preprint quant-ph/0410184 (2004).

    Args:
        ctrl (qubit): The control qubit.
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg += a_reg mod 2^n.

    """
    cx_ladder = CXLadderLinear()
    ancilla = qubit()

    if n == 1:
        toffoli(ctrl, a_reg[0], b_reg[0])
        discard(ancilla)
        return

    if n == 2:
        # Boundary UMAJ†: carry-in is known to be 0 and ancilla is clean.
        cx(a_reg[0], b_reg[0])
        temp_and_compute(a_reg[0], b_reg[0], ancilla)

        with dagger:
            _cntrl_maj_umaj_bottom_mod_gate(ctrl, a_reg[1], b_reg[1], ancilla)

        # Boundary controlled MAJ†: uncompute the clean ancilla.
        temp_and_compute(a_reg[0], b_reg[0], ancilla)
        toffoli(ctrl, a_reg[0], b_reg[0])
        cx(a_reg[0], b_reg[0])
        discard(ancilla)
        return

    toffoli(ctrl, a_reg[0], b_reg[0])
    transversal(cx, a_reg, b_reg, 1, comptime(n - 1))

    temp_and_compute(a_reg[0], b_reg[0], ancilla)

    # Forward n-2 CX ladder
    cx(a_reg[1], ancilla)
    cx_p, cx_q, cx_s = _unsafe_array_borrow_slice(
        a_reg, comptime(min(1, n - 1)), comptime(max(0, n - 2)), 1
    )
    cx_ladder.descending_dagger(cx_q)
    _unsafe_array_unborrow_slice(a_reg, cx_p, cx_q, cx_s)

    # Reverse n-2 Toffoli ladders
    toffoli(ctrl, ancilla, b_reg[1])
    toffoli(ancilla, b_reg[1], a_reg[1])
    for i in range(n - 4, -1, -1):
        toffoli(ctrl, a_reg[n - i - 3], b_reg[n - i - 2])
        toffoli(a_reg[n - i - 3], b_reg[n - i - 2], a_reg[n - i - 2])

    cx(a_reg[n - 2], a_reg[n - 1])
    toffoli(ctrl, a_reg[n - 1], b_reg[n - 1])
    cx(a_reg[n - 2], a_reg[n - 1])

    # Reverse n-2 Toffoli ladder
    for i in range(n - 4, -1, -1):
        toffoli(a_reg[i + 1], b_reg[i + 2], a_reg[i + 2])
    toffoli(ancilla, b_reg[1], a_reg[1])

    # Reverse n-2 CX ladder
    cx_p, cx_q, cx_s = _unsafe_array_borrow_slice(
        a_reg, comptime(min(1, n - 1)), comptime(max(0, n - 2)), 1
    )
    cx_ladder.descending(cx_q)
    _unsafe_array_unborrow_slice(a_reg, cx_p, cx_q, cx_s)
    cx(a_reg[1], ancilla)

    temp_and_uncompute(a_reg[0], b_reg[0], ancilla)

    transversal(cx, a_reg, b_reg, 1, comptime(n - 1))

    discard(ancilla)


@guppy
@no_type_check
def cntrl_adder_ripple_cuccaro_carry_out[n: nat](
    ctrl: qubit,
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
    carry_out: qubit,
) -> None:
    """Apply a controlled optimized Cuccaro ripple-carry adder.

    The circuit implements the transformation:

    ``|ctrl>|a>|b>|0> -> |ctrl>|a>|b + ctrl * a mod 2^n>|ctrl * carry_out>``

    The circuit requires one ancilla qubit. The circuit's depth depends on the supplied
    ladder implementations.

    Reference:
    Cuccaro, Steven A., et al. "A new quantum ripple-carry addition
    circuit." arXiv preprint quant-ph/0410184 (2004).

    Args:
        ctrl (qubit): The control qubit.
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg += a_reg mod 2^n.
        carry_out (qubit): Qubit that is flipped if the addition produces a final carry
            out.

    """
    cx_ladder = CXLadderLinear()
    ancilla = qubit()

    if n == 1:
        temp_and_compute(a_reg[0], b_reg[0], ancilla)
        temp_and_compute(ctrl, ancilla, carry_out)
        temp_and_uncompute(a_reg[0], b_reg[0], ancilla)
        toffoli(ctrl, a_reg[0], b_reg[0])
        discard(ancilla)
        return

    if n == 2:
        # Boundary MAJ: carry-in is known to be 0 and ancilla is clean.
        temp_and_compute(a_reg[0], b_reg[0], ancilla)

        _majority_gate(a_reg[1], b_reg[1], ancilla)
        temp_and_compute(ctrl, a_reg[1], carry_out)
        _cntrl_unmajority_gate(ctrl, a_reg[1], b_reg[1], ancilla)

        # Boundary controlled-UMAJ: uncompute the clean ancilla.
        temp_and_uncompute(a_reg[0], b_reg[0], ancilla)
        toffoli(ctrl, a_reg[0], b_reg[0])
        discard(ancilla)
        return

    # Boundary MAJ: carry-in is known to be 0 and ancilla is clean.
    temp_and_compute(a_reg[0], b_reg[0], ancilla)

    transversal(cx, a_reg, b_reg, 1, n)

    # Forward n-1 CX ladder
    cx(a_reg[1], ancilla)
    cx_p, cx_q, cx_s = _unsafe_array_borrow_slice(a_reg, 1, comptime(n - 1), 0)
    cx_ladder.descending_dagger(cx_q)
    _unsafe_array_unborrow_slice(a_reg, cx_p, cx_q, cx_s)

    # Forward n-1 Toffoli ladder
    toffoli(ancilla, b_reg[1], a_reg[1])
    for i in range(n - 2):
        toffoli(a_reg[i + 1], b_reg[i + 2], a_reg[i + 2])

    temp_and_compute(ctrl, a_reg[n - 1], carry_out)

    # Reverse n-1 Toffoli ladders
    for i in range(n - 2):
        toffoli(a_reg[n - i - 2], b_reg[n - i - 1], a_reg[n - i - 1])
        toffoli(ctrl, a_reg[n - i - 2], b_reg[n - i - 1])
    toffoli(ancilla, b_reg[1], a_reg[1])
    toffoli(ctrl, ancilla, b_reg[1])

    # Reverse n-1 CX ladder
    cx_p, cx_q, cx_s = _unsafe_array_borrow_slice(a_reg, 1, comptime(n - 1), 0)
    cx_ladder.descending(cx_q)
    _unsafe_array_unborrow_slice(a_reg, cx_p, cx_q, cx_s)
    cx(a_reg[1], ancilla)

    temp_and_uncompute(a_reg[0], b_reg[0], ancilla)

    transversal(cx, a_reg, b_reg, 1, n)
    toffoli(ctrl, a_reg[0], b_reg[0])

    discard(ancilla)


@guppy
@no_type_check
def _cntrl_adder_ripple_cuccaro_carry_out_dagger_impl[n: nat](
    ctrl: qubit,
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
    carry_out: qubit,
    uncompute_carry_out: bool,
) -> None:
    """Apply the inverse of the controlled optimized Cuccaro ripple-carry adder.

    The circuit implements the transformation:

    ``|ctrl>|a>|b>|0> -> |ctrl>|a>|b + ctrl * a mod 2^n>|ctrl * carry_out>``

    The circuit requires one ancilla qubit. The circuit's depth depends on the supplied
    ladder implementations.

    Reference:
    Cuccaro, Steven A., et al. "A new quantum ripple-carry addition
    circuit." arXiv preprint quant-ph/0410184 (2004).

    Args:
        ctrl (qubit): The control qubit.
        a_reg (array[qubit, n]): The addend register.
        b_reg (array[qubit, n]): The target register.
            Modified in-place: b_reg += a_reg mod 2^n.
        carry_out (qubit): Qubit that is flipped if the addition produces a final carry
            out.
        uncompute_carry_out: If `True`, uncompute the carry-out already stored in
            `carry_out`. If `False`, `carry_out` must be a clean ancilla and is
            populated with the carry-out produced by the inverse circuit.

    """
    cx_ladder = CXLadderLinear()
    ancilla = qubit()

    if n == 1:
        toffoli(ctrl, a_reg[0], b_reg[0])
        temp_and_compute(a_reg[0], b_reg[0], ancilla)
        temp_and_compute(ctrl, ancilla, carry_out)
        temp_and_uncompute(a_reg[0], b_reg[0], ancilla)
        discard(ancilla)
        return

    if n == 2:
        # Boundary UMAJ†: carry-in is known to be 0 and ancilla is clean.
        cx(a_reg[0], b_reg[0])
        temp_and_compute(a_reg[0], b_reg[0], ancilla)

        with dagger:
            _unmajority_gate(a_reg[1], b_reg[1], ancilla)
        if uncompute_carry_out:
            temp_and_uncompute(ctrl, a_reg[1], carry_out)
        else:
            temp_and_compute(ctrl, a_reg[1], carry_out)
        _cntrl_majority_dagger_gate(ctrl, a_reg[1], b_reg[1], ancilla)

        # Boundary controlled-MAJ†: uncompute the clean ancilla.
        temp_and_compute(a_reg[0], b_reg[0], ancilla)
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
    if uncompute_carry_out:
        temp_and_uncompute(ctrl, a_reg[n - 1], carry_out)
    else:
        temp_and_compute(ctrl, a_reg[n - 1], carry_out)
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
def cntrl_adder_ripple_cuccaro_carry_out_dagger[n: nat](
    ctrl: qubit,
    a_reg: array[qubit, n],
    b_reg: array[qubit, n],
    carry_out: qubit,
) -> None:
    """Apply the inverse of the controlled Cuccaro ripple-carry adder.

    Computes ``b - a`` in place on ``b_reg`` conditioned on the control qubit while
    restoring ``a_reg``. The supplied ``borrow_out`` qubit is treated as the carry-out
    qubit from the forward adder and is uncomputed by the inverse operation.

    Transformation::

        |ctrl>|a>|b + ctrl * a mod 2^n>|ctrl * carry_out> -> |ctrl>|a>|b>|0>

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
        carry_out (qubit): Carry/borrow qubit to uncompute.

    """
    return _cntrl_adder_ripple_cuccaro_carry_out_dagger_impl(
        ctrl, a_reg, b_reg, carry_out, True
    )
