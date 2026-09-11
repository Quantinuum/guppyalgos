"""Tests for ripple-carry modular multipliers."""

from typing import no_type_check

import pytest
from guppylang_internals.error import GuppyError

from guppylang import guppy
from guppylang.std.builtins import array, output, nat
from guppylang.std.quantum import (
    collect_measurements,
    discard,
    discard_array,
    h,
    measure,
    measure_array,
    qubit,
    x,
)

from guppyalgos.primitives.arithmetic import (
    cntrl_multiplier_ripple_gidney_mod_in_place,
    cntrl_multiplier_ripple_gidney_mod,
    multiplier_ripple_gidney_mod_in_place,
    multiplier_ripple_gidney_mod,
)
from guppyalgos.utils import apply_bitstring, int_to_bits, qarray


@pytest.mark.parametrize(("n", "a", "b"), [(2, 1, 1), (2, 2, 3), (3, 3, 2), (4, 5, 7)])
def test_multiplier_ripple_gidney_mod[n: nat](n: int, a: int, b: int) -> None:
    """Checks that the multiplier correctly computes a * b modulo 2**n."""
    a_bits = int_to_bits(a, n)
    b_bits = int_to_bits(b, n)
    expected = (a * b) % 2**n
    a_bit_array = array(*a_bits)
    b_bit_array = array(*b_bits)

    @guppy
    @no_type_check
    def main() -> None:
        a_reg = qarray(n)
        apply_bitstring(a_reg, a_bit_array)
        multiplier = qarray(n)
        apply_bitstring(multiplier, b_bit_array)
        product = qarray(n)

        multiplier_ripple_gidney_mod(a_reg, multiplier, product)

        output("a_meas", collect_measurements(measure_array(a_reg)))
        output("mult_meas", collect_measurements(measure_array(multiplier)))
        output("prod_meas", collect_measurements(measure_array(product)))

    res = main.emulator(n_qubits=4 * n).run()
    result = res.results[0].as_dict()
    assert result["a_meas"] == a_bits
    assert result["mult_meas"] == b_bits
    assert result["prod_meas"] == int_to_bits(expected, n)


@pytest.mark.parametrize(("n", "a", "b"), [(2, 1, 1), (2, 2, 3), (3, 3, 2), (4, 5, 7)])
@pytest.mark.parametrize("ctrl_active", [False, True])
def test_cntrl_multiplier_ripple_gidney_mod[n: nat](
    n: int,
    a: int,
    b: int,
    ctrl_active: bool,
) -> None:
    """Check the controlled multiplier correctly computes the product.

    Checks that the controlled multiplier correctly computes a * b modulo 2**n when the
    control qubit is active, and leaves the product register unchanged otherwise.
    """
    a_bits = int_to_bits(a, n)
    b_bits = int_to_bits(b, n)
    expected = 0 if not ctrl_active else (a * b) % 2**n
    a_bit_array = array(*a_bits)
    b_bit_array = array(*b_bits)

    @guppy
    @no_type_check
    def main() -> None:
        a_reg = qarray(n)
        apply_bitstring(a_reg, a_bit_array)
        multiplier = qarray(n)
        apply_bitstring(multiplier, b_bit_array)
        product = qarray(n)

        ctrl = qubit()
        if ctrl_active:
            x(ctrl)

        cntrl_multiplier_ripple_gidney_mod(ctrl, a_reg, multiplier, product)

        output("a_meas", collect_measurements(measure_array(a_reg)))
        output("mult_meas", collect_measurements(measure_array(multiplier)))
        output("prod_meas", collect_measurements(measure_array(product)))

        discard(ctrl)

    res = main.emulator(n_qubits=4 * n + 2).run()
    result = res.results[0].as_dict()
    assert result["a_meas"] == a_bits
    assert result["mult_meas"] == b_bits
    assert result["prod_meas"] == int_to_bits(expected, n)


@pytest.mark.parametrize(("n", "a", "b"), [(2, 1, 1), (2, 2, 3), (3, 3, 2), (4, 5, 7)])
def test_multiplier_ripple_gidney_mod_superposition[n: nat](
    n: int,
    a: int,
    b: int,
) -> None:
    """Check that the multiplier correctly handles superposition states.

    The uniform superposition on ``n`` qubits is invariant under addition modulo
    ``2**n``. Since the multiplier effects the transformation
    ``|a>|b>|p> -> |a>|b>|p + a*b mod 2**n>``, if ``|p>`` is initially in the uniform
    superposition, it should remain unchanged after the multiplication.
    """
    a_bits = int_to_bits(a, n)
    b_bits = int_to_bits(b, n)
    a_bit_array = array(*a_bits)
    b_bit_array = array(*b_bits)

    @guppy
    @no_type_check
    def main() -> None:
        a_reg = qarray(n)
        apply_bitstring(a_reg, a_bit_array)
        multiplier = qarray(n)
        apply_bitstring(multiplier, b_bit_array)
        product = qarray(n)

        for i in range(n):
            h(product[i])

        multiplier_ripple_gidney_mod(a_reg, multiplier, product)

        for i in range(n):
            h(product[i])

        output("a_meas", collect_measurements(measure_array(a_reg)))
        output("mult_meas", collect_measurements(measure_array(multiplier)))
        output("prod_meas", collect_measurements(measure_array(product)))

    res = main.emulator(n_qubits=4 * n).run()
    result = res.results[0].as_dict()
    assert result["a_meas"] == a_bits
    assert result["mult_meas"] == b_bits
    assert result["prod_meas"] == [False] * n


@pytest.mark.parametrize("n", [2, 3, 4])
def test_multiplier_ripple_gidney_mod_full_superposition[n: nat](n: int) -> None:
    """Check that the multiplier preserves a uniform superposition on all registers."""

    @guppy
    @no_type_check
    def main() -> None:
        a_reg = qarray(n)
        multiplier = qarray(n)
        product = qarray(n)

        for i in range(n):
            h(a_reg[i])
            h(multiplier[i])
            h(product[i])

        multiplier_ripple_gidney_mod(a_reg, multiplier, product)

        for i in range(n):
            h(a_reg[i])
            h(multiplier[i])
            h(product[i])

        output("a_meas", collect_measurements(measure_array(a_reg)))
        output("mult_meas", collect_measurements(measure_array(multiplier)))
        output("prod_meas", collect_measurements(measure_array(product)))

    res = main.emulator(n_qubits=4 * n).run()
    result = res.results[0].as_dict()
    zero_bits = [False] * n
    assert result["a_meas"] == zero_bits
    assert result["mult_meas"] == zero_bits
    assert result["prod_meas"] == zero_bits


@pytest.mark.parametrize(("n", "a", "b"), [(2, 2, 1), (3, 3, 3), (4, 5, 7)])
def test_in_place_multiplier_ripple_gidney_mod[n: nat](
    n: int,
    a: int,
    b: int,
) -> None:
    """Check in-place multiplication by an odd classical constant."""
    a_bits = int_to_bits(a, n)
    a_bit_array = array(*a_bits)

    @guppy
    @no_type_check
    def main() -> None:
        a_reg = qarray(n)
        apply_bitstring(a_reg, a_bit_array)

        multiplier_ripple_gidney_mod_in_place(a_reg, b)

        output("a_meas", collect_measurements(measure_array(a_reg)))

    res = main.emulator(n_qubits=4 * n).run()
    result = res.results[0].as_dict()
    assert result["a_meas"] == int_to_bits((a * b) % 2**n, n)


@pytest.mark.parametrize(("n", "b"), [(2, 3), (3, 5), (4, 7)])
def test_in_place_multiplier_ripple_gidney_mod_superposition[n: nat](
    n: int,
    b: int,
) -> None:
    """Check phases and ancilla cleanup on a uniform superposition."""

    @guppy
    @no_type_check
    def main() -> None:
        a_reg = qarray(n)
        for i in range(n):
            h(a_reg[i])

        multiplier_ripple_gidney_mod_in_place(a_reg, b)

        for i in range(n):
            h(a_reg[i])
        output("a_meas", collect_measurements(measure_array(a_reg)))

    res = main.emulator(n_qubits=4 * n).run()
    result = res.results[0].as_dict()
    assert result["a_meas"] == [False] * n


@pytest.mark.parametrize(("n", "a", "b"), [(1, 1, 3), (2, 2, 1), (3, 3, 3), (4, 5, 7)])
@pytest.mark.parametrize("ctrl_active", [False, True])
def test_cntrl_in_place_multiplier_ripple_gidney_mod[n: nat](
    n: int,
    a: int,
    b: int,
    ctrl_active: bool,
) -> None:
    """Check controlled in-place multiplication by an odd classical constant."""
    a_bits = int_to_bits(a, n)
    expected = (a * b) % 2**n if ctrl_active else a
    a_bit_array = array(*a_bits)

    @guppy
    @no_type_check
    def main() -> None:
        a_reg = qarray(n)
        apply_bitstring(a_reg, a_bit_array)
        ctrl = qubit()
        if ctrl_active:
            x(ctrl)

        cntrl_multiplier_ripple_gidney_mod_in_place(ctrl, a_reg, b)

        output("a_meas", collect_measurements(measure_array(a_reg)))
        discard(ctrl)

    res = main.emulator(n_qubits=4 * n + 2).run()
    result = res.results[0].as_dict()
    assert result["a_meas"] == int_to_bits(expected, n)


@pytest.mark.parametrize(("n", "b"), [(1, 3), (2, 3), (3, 5), (4, 7)])
def test_cntrl_in_place_multiplier_ripple_gidney_mod_superposition[n: nat](
    n: int,
    b: int,
) -> None:
    """Check coherent control, phases, and ancilla cleanup in superposition."""

    @guppy
    @no_type_check
    def main() -> None:
        a_reg = qarray(n)
        ctrl = qubit()
        h(ctrl)
        for i in range(n):
            h(a_reg[i])

        cntrl_multiplier_ripple_gidney_mod_in_place(ctrl, a_reg, b)

        h(ctrl)
        for i in range(n):
            h(a_reg[i])
        output("a_meas", collect_measurements(measure_array(a_reg)))
        output("ctrl_meas", measure(ctrl).read())

    res = main.emulator(n_qubits=4 * n + 2).run()
    result = res.results[0].as_dict()
    assert result["a_meas"] == [False] * n
    assert result["ctrl_meas"] == 0


@pytest.mark.parametrize(
    "multiplier",
    [
        multiplier_ripple_gidney_mod_in_place,
        cntrl_multiplier_ripple_gidney_mod_in_place,
    ],
)
@pytest.mark.parametrize(
    ("b", "error_message"),
    [
        (0, "requires an odd multiplier"),
        (2, "requires an odd multiplier"),
        (-1, "requires a non-negative multiplier"),
        (-3, "requires a non-negative multiplier"),
    ],
)
def test_in_place_multiplier_rejects_invalid_classical_multiplier(
    multiplier, b: int, error_message: str
) -> None:
    """Check that in-place multipliers reject even and negative constants."""
    if multiplier is cntrl_multiplier_ripple_gidney_mod_in_place:

        @guppy
        @no_type_check
        def main() -> None:
            a_reg = qarray(2)
            ctrl = qubit()
            multiplier(ctrl, a_reg, b)
            discard(ctrl)
            discard_array(a_reg)

    else:

        @guppy
        @no_type_check
        def main() -> None:
            a_reg = qarray(2)
            multiplier(a_reg, b)
            discard_array(a_reg)

    with pytest.raises(GuppyError, match=error_message):
        main.emulator(n_qubits=1).run()
