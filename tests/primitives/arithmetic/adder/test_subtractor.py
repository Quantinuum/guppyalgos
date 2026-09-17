"""Tests for subtractors."""

from typing import no_type_check
from collections.abc import Callable

import pytest
import numpy as np

from guppylang.decorator import guppy
from guppylang.std.builtins import array, output, nat
from guppylang.std.debug import state_output
from guppylang.std.quantum import (
    collect_measurements,
    measure,
    measure_array,
    discard,
    qubit,
    x,
)
from guppylang.defs import GuppyFunctionDefinition

from selene_sim import Quest

from guppyalgos.primitives.arithmetic import (
    subtractor_ripple_cuccaro_carry_out,
    subtractor_ripple_cuccaro_mod,
    subtractor_ripple_gidney_carry_out,
    subtractor_ripple_gidney_mod,
    cntrl_subtractor_ripple_cuccaro_carry_out,
    cntrl_subtractor_ripple_cuccaro_mod,
    cntrl_subtractor_ripple_gidney_carry_out,
    cntrl_subtractor_ripple_gidney_mod,
)
from guppyalgos.utils import apply_bitstring, int_to_bits, qarray
from guppyalgos.testing import project_state_onto_bitstring


@pytest.mark.parametrize(
    ("n", "cases"),
    [
        (2, [(0, 0), (1, 0), (3, 1), (1, 3)]),
        (3, [(0, 7), (7, 0), (5, 2), (2, 5)]),
        (4, [(0, 15), (15, 0), (8, 8), (7, 9)]),
        (5, [(0, 31), (31, 0), (20, 10), (10, 25)]),
        (6, [(32, 5), (5, 32)]),
    ],
)
@pytest.mark.parametrize(
    ("subtractor", "num_ancilla_fn"),
    [
        (subtractor_ripple_cuccaro_carry_out, lambda n: 1),
        (subtractor_ripple_gidney_carry_out, lambda n: n - 1),
    ],
)
def test_subtraction_carry_out[n: nat](
    n: int,
    cases: list[tuple[int, int]],
    subtractor: GuppyFunctionDefinition[
        [array[qubit, n], array[qubit, n], qubit], None
    ],
    num_ancilla_fn: Callable[[int], int],
) -> None:
    """Test ripple carry subtraction circuit.

    Tests output subtraction and carry out bit.
    """
    n_qubits = 2 * n + 1 + num_ancilla_fn(n)

    @guppy
    @no_type_check
    def main(_a_bit_array: array[bool, n], _b_bit_array: array[bool, n]) -> None:
        """Run the main test function."""
        a_reg = qarray(n)
        b_reg = qarray(n)
        apply_bitstring(a_reg, _a_bit_array)
        apply_bitstring(b_reg, _b_bit_array)
        comp = qubit()
        subtractor(a_reg, b_reg, comp)

        state_output("b_reg", b_reg)

        output("comp", measure(comp).read())
        output("b_meas", collect_measurements(measure_array(b_reg)))
        output("a_meas", collect_measurements(measure_array(a_reg)))

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=n_qubits)
    for a, b in cases:
        a_bits = int_to_bits(a, n)
        b_bits = int_to_bits(b, n)

        res = emulator.run(_a_bit_array=a_bits, _b_bit_array=b_bits)
        assert (res.results[0].as_dict()["comp"]) == (b < a)
        assert res.results[0].as_dict()["a_meas"] == int_to_bits(a, n)
        assert res.results[0].as_dict()["b_meas"] == int_to_bits((b - a) % 2**n, n)

        states = Quest.extract_states_dict(res.results[0].entries)

        b_proj = project_state_onto_bitstring(
            states["b_reg"], int_to_bits((b - a) % 2**n, n)
        )
        assert np.allclose(b_proj.probability, 1.0)


@pytest.mark.parametrize(
    ("n", "cases"),
    [
        (2, [(0, 0), (1, 0), (3, 1), (1, 3)]),
        (3, [(0, 7), (7, 0), (5, 2), (2, 5)]),
        (4, [(0, 15), (15, 0), (8, 8), (7, 9)]),
        (5, [(0, 31), (31, 0), (20, 10), (10, 25)]),
        (6, [(32, 5), (5, 32)]),
    ],
)
@pytest.mark.parametrize(
    ("subtractor", "num_ancilla_fn"),
    [
        (subtractor_ripple_cuccaro_mod, lambda n: 1),
        (subtractor_ripple_gidney_mod, lambda n: n - 1),
    ],
)
def test_subtraction_mod[n: nat](
    n: int,
    cases: list[tuple[int, int]],
    subtractor: GuppyFunctionDefinition[[array[qubit, n], array[qubit, n]], None],
    num_ancilla_fn: Callable[[int], int],
) -> None:
    """Test ripple carry modular subtraction circuit."""
    n_qubits = 2 * n + num_ancilla_fn(n)

    @guppy
    @no_type_check
    def main(_a_bit_array: array[bool, n], _b_bit_array: array[bool, n]) -> None:
        """Run the main test function."""
        a_reg = qarray(n)
        b_reg = qarray(n)
        apply_bitstring(a_reg, _a_bit_array)
        apply_bitstring(b_reg, _b_bit_array)
        comp = qubit()
        discard(comp)

        subtractor(a_reg, b_reg)

        state_output("a_reg", a_reg)
        state_output("b_reg", b_reg)

        output("a_meas", collect_measurements(measure_array(a_reg)))
        output("b_meas", collect_measurements(measure_array(b_reg)))

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=n_qubits)
    for a, b in cases:
        a_bits = int_to_bits(a, n)
        b_bits = int_to_bits(b, n)

        res = emulator.run(_a_bit_array=a_bits, _b_bit_array=b_bits)
        assert res.results[0].as_dict()["a_meas"] == int_to_bits(a, n)
        assert res.results[0].as_dict()["b_meas"] == int_to_bits((b - a) % 2**n, n)

        states = Quest.extract_states_dict(res.results[0].entries)

        b_proj = project_state_onto_bitstring(
            states["b_reg"], int_to_bits((b - a) % 2**n, n)
        )
        assert np.allclose(b_proj.probability, 1.0)

        a_proj = project_state_onto_bitstring(states["a_reg"], int_to_bits(a, n))
        assert np.allclose(a_proj.probability, 1.0)


@pytest.mark.parametrize(
    ("n", "cases"),
    [
        (2, [(0, 0), (1, 0), (3, 1), (1, 3)]),
        (3, [(0, 7), (7, 0), (5, 2), (2, 5)]),
        (4, [(0, 15), (15, 0), (8, 8), (7, 9)]),
        (5, [(0, 31), (31, 0), (20, 10), (10, 25)]),
        (6, [(32, 5), (5, 32)]),
    ],
)
@pytest.mark.parametrize(
    ("controlled_subtractor", "num_ancilla_fn"),
    [
        (
            cntrl_subtractor_ripple_cuccaro_carry_out,
            lambda n: 1,
        ),
        (
            cntrl_subtractor_ripple_gidney_carry_out,
            lambda n: n,
        ),
    ],
)
def test_cntrl_subtraction_carry_out[n: nat](
    n: int,
    cases: list[tuple[int, int]],
    controlled_subtractor: GuppyFunctionDefinition[
        [qubit, array[qubit, n], array[qubit, n], qubit],
        None,
    ],
    num_ancilla_fn: Callable[[int], int],
) -> None:
    """Test controlled ripple-carry subtraction and comparison bit."""
    n_qubits = 2 * n + 2 + num_ancilla_fn(n)

    @guppy
    @no_type_check
    def main(
        _a_bit_array: array[bool, n], _b_bit_array: array[bool, n], ctrl_active: bool
    ) -> None:
        a_reg = qarray(n)
        b_reg = qarray(n)
        apply_bitstring(a_reg, _a_bit_array)
        apply_bitstring(b_reg, _b_bit_array)
        comp = qubit()

        ctrl = qubit()
        if ctrl_active:
            x(ctrl)

        controlled_subtractor(ctrl, a_reg, b_reg, comp)

        state_output("b_reg", b_reg)
        state_output("comp", comp)

        output("ctrl", measure(ctrl).read())
        output("comp_meas", measure(comp).read())
        output("b_meas", collect_measurements(measure_array(b_reg)))
        output("a_meas", collect_measurements(measure_array(a_reg)))

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=n_qubits)
    for ctrl_active in [False, True]:
        for a, b in cases:
            a_bits = int_to_bits(a, n)

            b_bits = int_to_bits(b, n)

            expected_b = (b - a) % 2**n if ctrl_active else b
            expected_comp = (b < a) if ctrl_active else False

            res = emulator.run(
                _a_bit_array=a_bits, _b_bit_array=b_bits, ctrl_active=ctrl_active
            )
            result = res.results[0]

            assert result.as_dict()["ctrl"] == ctrl_active
            assert result.as_dict()["comp_meas"] == expected_comp
            assert result.as_dict()["a_meas"] == a_bits
            assert result.as_dict()["b_meas"] == int_to_bits(expected_b, n)

            states = Quest.extract_states_dict(result.entries)

            b_proj = project_state_onto_bitstring(
                states["b_reg"], int_to_bits(expected_b, n)
            )
            assert b_proj.probability == pytest.approx(1.0)

            comp_proj = project_state_onto_bitstring(states["comp"], [expected_comp])
            assert comp_proj.probability == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("n", "cases"),
    [
        (2, [(0, 0), (1, 0), (3, 1), (1, 3)]),
        (3, [(0, 7), (7, 0), (5, 2), (2, 5)]),
        (4, [(0, 15), (15, 0), (8, 8), (7, 9)]),
        (5, [(0, 31), (31, 0), (20, 10), (10, 25)]),
        (6, [(32, 5), (5, 32)]),
    ],
)
@pytest.mark.parametrize(
    ("controlled_subtractor", "num_ancilla_fn"),
    [
        (
            cntrl_subtractor_ripple_cuccaro_mod,
            lambda n: 1,
        ),
        (
            cntrl_subtractor_ripple_gidney_mod,
            lambda n: n,
        ),
    ],
)
def test_cntrl_subtraction_mod[n: nat](
    n: int,
    cases: list[tuple[int, int]],
    controlled_subtractor: GuppyFunctionDefinition[
        [qubit, array[qubit, n], array[qubit, n]],
        None,
    ],
    num_ancilla_fn: Callable[[int], int],
) -> None:
    """Test controlled ripple-carry modular subtraction."""
    n_qubits = 2 * n + 1 + num_ancilla_fn(n)

    @guppy
    @no_type_check
    def main(
        _a_bit_array: array[bool, n], _b_bit_array: array[bool, n], ctrl_active: bool
    ) -> None:
        a_reg = qarray(n)
        b_reg = qarray(n)
        apply_bitstring(a_reg, _a_bit_array)
        apply_bitstring(b_reg, _b_bit_array)
        comp = qubit()
        discard(comp)

        ctrl = qubit()
        if ctrl_active:
            x(ctrl)

        controlled_subtractor(ctrl, a_reg, b_reg)

        state_output("ctrl", ctrl)
        state_output("a_reg", a_reg)
        state_output("b_reg", b_reg)

        output("ctrl_meas", measure(ctrl).read())
        output("a_meas", collect_measurements(measure_array(a_reg)))
        output("b_meas", collect_measurements(measure_array(b_reg)))

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=n_qubits)
    for ctrl_active in [False, True]:
        for a, b in cases:
            a_bits = int_to_bits(a, n)

            b_bits = int_to_bits(b, n)

            expected_b = (b - a) % 2**n if ctrl_active else b

            res = emulator.run(
                _a_bit_array=a_bits, _b_bit_array=b_bits, ctrl_active=ctrl_active
            )
            result = res.results[0]

            assert result.as_dict()["ctrl_meas"] == ctrl_active
            assert result.as_dict()["a_meas"] == a_bits
            assert result.as_dict()["b_meas"] == int_to_bits(expected_b, n)

            states = Quest.extract_states_dict(result.entries)

            ctrl_proj = project_state_onto_bitstring(states["ctrl"], [ctrl_active])
            assert ctrl_proj.probability == pytest.approx(1.0)

            a_proj = project_state_onto_bitstring(states["a_reg"], a_bits)
            assert a_proj.probability == pytest.approx(1.0)

            b_proj = project_state_onto_bitstring(
                states["b_reg"], int_to_bits(expected_b, n)
            )
            assert b_proj.probability == pytest.approx(1.0)
