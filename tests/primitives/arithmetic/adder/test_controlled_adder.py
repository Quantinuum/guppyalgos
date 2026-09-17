"""Tests for the controlled adders."""

from typing import no_type_check
from collections.abc import Callable

import numpy as np
import pytest

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, output, nat
from guppylang.std.debug import state_result, state_output
from guppylang.std.quantum import (
    collect_measurements,
    discard,
    discard_array,
    h,
    measure_array,
    qubit,
    x,
)
from selene_sim import Quest

from guppyalgos.primitives.arithmetic import (
    cntrl_adder_ripple_gidney_mod,
    cntrl_adder_ripple_gidney_carry_out,
    cntrl_adder_ripple_cuccaro_mod,
    cntrl_adder_ripple_cuccaro_carry_out,
    cntrl_adder_ripple_cuccaro_carry_out_dagger,
    cntrl_adder_ripple_cuccaro_mod_dagger,
    cntrl_adder_ripple_gidney_mod_dagger,
    cntrl_adder_ripple_gidney_carry_out_dagger,
)
from guppyalgos.utils import qarray, int_to_bits, apply_bitstring
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_total_state_on_only_specified_registers,
    project_state_onto_bitstring,
)


@pytest.mark.parametrize(
    ("n", "cases"),
    [
        (2, [(3, 1), (1, 2), (3, 3)]),
        (3, [(3, 4), (5, 2), (7, 7)]),
        (4, [(3, 7), (9, 4), (15, 15)]),
        (5, [(5, 12), (18, 6), (31, 31)]),
    ],
)
@pytest.mark.parametrize(
    ("controlled_adder", "num_ancilla_fn"),
    [
        (cntrl_adder_ripple_gidney_mod, lambda n: n),
        (cntrl_adder_ripple_cuccaro_mod, lambda n: 1),
    ],
)
def test_cntrl_adder_mod[n: nat](
    n: int,
    cases: list[tuple[int, int]],
    controlled_adder: GuppyFunctionDefinition[
        [qubit, array[qubit, n], array[qubit, n]], None
    ],
    num_ancilla_fn: Callable[[int], int],
) -> None:
    """Check basis-state controlled addition by measuring the target register."""
    n_qubits = 2 * n + 1 + num_ancilla_fn(n)

    @guppy
    @no_type_check
    def main(
        _a_bit_array: array[bool, n], _b_bit_array: array[bool, n], ctrl_active: bool
    ) -> None:
        """Run the main test function."""
        a_reg = qarray(n)
        b_reg = qarray(n)
        apply_bitstring(a_reg, _a_bit_array)
        apply_bitstring(b_reg, _b_bit_array)
        carry_out = qubit()
        discard(carry_out)

        ctrl = qubit()
        if ctrl_active:
            x(ctrl)

        controlled_adder(ctrl, a_reg, b_reg)

        discard(ctrl)

        output("a_meas", collect_measurements(measure_array(a_reg)))
        output("b_meas", collect_measurements(measure_array(b_reg)))

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=n_qubits)
    for ctrl_active in [True, False]:
        for a, b in cases:
            a_bits = int_to_bits(a, n)

            b_bits = int_to_bits(b, n)

            expected_bits = int_to_bits((a + b) % 2**n, n) if ctrl_active else b_bits

            res = emulator.run(
                _a_bit_array=a_bits, _b_bit_array=b_bits, ctrl_active=ctrl_active
            )
            assert res.results[0].as_dict()["b_meas"] == expected_bits
            assert res.results[0].as_dict()["a_meas"] == a_bits


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize(
    ("controlled_adder", "num_ancilla_fn"),
    [
        (cntrl_adder_ripple_gidney_mod, lambda n: n),
        (cntrl_adder_ripple_cuccaro_mod, lambda n: 1),
    ],
)
def test_cntrl_adder_mod_statevector_superposition[n: nat](
    n: int,
    controlled_adder: GuppyFunctionDefinition[
        [qubit, array[qubit, n], array[qubit, n]], None
    ],
    num_ancilla_fn: Callable[[int], int],
) -> None:
    """Check controlled addition on a superposition, including relative phases.

    The input is

        ctrl = |+>
        a    = |0>^{n-1} |+>
        b    = |0>^{n-1} |1>

    so the a register is in an equal superposition of the integer values
    0 and 1. Together with ctrl = |+>, this gives four equally weighted
    branches corresponding to ctrl, a in {0, 1}. The controlled adder should map each
    branch to

        |ctrl, a, 1 + ctrl * a mod 2**n>,

    without introducing branch-dependent phases or leaking amplitude into
    ancillary work qubits.

    The expected logical state is stored as a flat statevector with qubit
    positions ordered as

        0              -> ctrl
        1 .. n         -> a[0] .. a[n - 1]
        n + 1 .. 2n    -> b[0] .. b[n - 1]

    Hence the basis-state index for |ctrl, a, b> is

        ctrl + 2 a + 2^{n + 1} b
    """
    n_qubits = 2 * n + 1 + num_ancilla_fn(n)

    @guppy
    @no_type_check
    def main() -> None:
        ctrl = qubit()
        h(ctrl)

        a_reg = qarray(n)
        h(a_reg[0])

        b_reg = qarray(n)
        x(b_reg[0])

        controlled_adder(ctrl, a_reg, b_reg)

        state_result("ctrl", ctrl)
        state_result("a", a_reg)
        state_result("b", b_reg)

        discard(ctrl)
        discard_array(a_reg)
        discard_array(b_reg)

    emulator_result = main.emulator(n_qubits=n_qubits).with_seed(5).run()
    states = Quest.extract_states_dict(emulator_result.results[0].entries)
    total_state, _ = get_total_state_on_only_specified_registers(
        states, ["ctrl", "a", "b"]
    )

    # Sanity-check the expected computational-basis support independently of
    # the flat statevector indexing below.
    for ctrl_bit in (0, 1):
        ctrl_proj = project_state_onto_bitstring(states["ctrl"], [bool(ctrl_bit)])
        assert ctrl_proj.probability == pytest.approx(0.5)

    for a_value in (0, 1):
        a_proj = project_state_onto_bitstring(states["a"], int_to_bits(a_value, n))
        assert a_proj.probability == pytest.approx(0.5)

    b_one_proj = project_state_onto_bitstring(states["b"], int_to_bits(1, n))
    assert b_one_proj.probability == pytest.approx(0.75)

    b_two_proj = project_state_onto_bitstring(states["b"], int_to_bits(2, n))
    assert b_two_proj.probability == pytest.approx(0.25)

    expected_state = np.zeros(2 ** (2 * n + 1), dtype=np.complex128)
    for ctrl_bit in (0, 1):
        for a_value in (0, 1):
            # The controlled adder should map each branch to
            # |ctrl, a, b> --> |ctrl, a, 1 + ctrl * a mod 2**n>
            b_value = (1 + ctrl_bit * a_value) % (2**n)

            # basis-state index for |ctrl, a, b> is ctrl + 2 a + 2^{n + 1} b
            index = ctrl_bit + (a_value << 1) + (b_value << (n + 1))
            expected_state[index] = 0.5

    assert_allclose_ignorephase(total_state.state, expected_state)


@pytest.mark.parametrize(
    ("n", "cases"),
    [
        (2, [(3, 1), (1, 2), (3, 3)]),
        (3, [(3, 4), (5, 2), (7, 7)]),
        (4, [(3, 7), (9, 4), (15, 15)]),
        (5, [(5, 12), (18, 6), (31, 31)]),
        (6, [(5, 32), (48, 6)]),
    ],
)
@pytest.mark.parametrize(
    ("controlled_adder", "num_ancilla_fn"),
    [
        (cntrl_adder_ripple_cuccaro_carry_out, lambda n: 1),
        (cntrl_adder_ripple_gidney_carry_out, lambda n: n),
    ],
)
def test_cntrl_adder_carry_out[n: nat](
    n: int,
    cases: list[tuple[int, int]],
    controlled_adder: GuppyFunctionDefinition[
        [qubit, array[qubit, n], array[qubit, n], qubit], None
    ],
    num_ancilla_fn: Callable[[int], int],
) -> None:
    """Check basis-state controlled addition by measuring the target register."""
    n_qubits = 2 * n + 2 + num_ancilla_fn(n)

    @guppy
    @no_type_check
    def main(
        _a_bit_array: array[bool, n], _b_bit_array: array[bool, n], ctrl_active: bool
    ) -> None:
        """Run the main test function."""
        a_reg = qarray(n)
        b_reg = qarray(n)
        apply_bitstring(a_reg, _a_bit_array)
        apply_bitstring(b_reg, _b_bit_array)
        carry_out = qubit()

        ctrl = qubit()
        if ctrl_active:
            x(ctrl)

        controlled_adder(ctrl, a_reg, b_reg, carry_out)

        state_output("b_reg", b_reg)
        state_output("carry_out", carry_out)

        discard(ctrl)
        discard(carry_out)

        output("a_meas", collect_measurements(measure_array(a_reg)))
        output("b_meas", collect_measurements(measure_array(b_reg)))

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=n_qubits)
    for ctrl_active in [True, False]:
        for a, b in cases:
            a_bits = int_to_bits(a, n)

            b_bits = int_to_bits(b, n)

            expected_bits = int_to_bits((a + b) % 2**n, n) if ctrl_active else b_bits
            expected_total = (a + b) if ctrl_active else b

            res = emulator.run(
                _a_bit_array=a_bits, _b_bit_array=b_bits, ctrl_active=ctrl_active
            )
            assert res.results[0].as_dict()["b_meas"] == expected_bits
            assert res.results[0].as_dict()["a_meas"] == a_bits

            states = Quest.extract_states_dict(res.results[0].entries)
            total_state, _ = get_total_state_on_only_specified_registers(
                states, ["b_reg", "carry_out"]
            )
            total_state = total_state.state
            assert total_state[expected_total] == 1


@pytest.mark.parametrize(
    ("n", "cases"),
    [
        (2, [(3, 1), (1, 2), (3, 3)]),
        (3, [(3, 4), (5, 2), (7, 7)]),
        (4, [(3, 7), (9, 4), (15, 15)]),
        (5, [(5, 12), (18, 6), (31, 31)]),
    ],
)
@pytest.mark.parametrize(
    ("controlled_adder", "controlled_adder_dagger", "num_ancilla_fn"),
    [
        (
            cntrl_adder_ripple_cuccaro_carry_out,
            cntrl_adder_ripple_cuccaro_carry_out_dagger,
            lambda n: 1,
        ),
        (
            cntrl_adder_ripple_gidney_carry_out,
            cntrl_adder_ripple_gidney_carry_out_dagger,
            lambda n: n,
        ),
    ],
)
def test_cntrl_addition_carry_out_dagger[n: nat](
    n: int,
    cases: list[tuple[int, int]],
    controlled_adder: GuppyFunctionDefinition[
        [qubit, array[qubit, n], array[qubit, n], qubit],
        None,
    ],
    controlled_adder_dagger: GuppyFunctionDefinition[
        [qubit, array[qubit, n], array[qubit, n], qubit],
        None,
    ],
    num_ancilla_fn: Callable[[int], int],
) -> None:
    """Test controlled ripple-carry dagger inverts correctly."""
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
        carry_out = qubit()

        ctrl = qubit()
        if ctrl_active:
            x(ctrl)

        controlled_adder(ctrl, a_reg, b_reg, carry_out)
        controlled_adder_dagger(ctrl, a_reg, b_reg, carry_out)

        if ctrl_active:
            x(ctrl)

        apply_bitstring(a_reg, _a_bit_array)
        apply_bitstring(b_reg, _b_bit_array)

        state_output("ctrl", ctrl)
        state_output("a_reg", a_reg)
        state_output("b_reg", b_reg)

        # carry_out was measurement-uncomputed, so its post-measurement
        # value is irrelevant
        discard(carry_out)

        discard(ctrl)
        discard_array(a_reg)
        discard_array(b_reg)

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=n_qubits).with_seed(42)
    for ctrl_active in [False, True]:
        for a, b in cases:
            a_bits = int_to_bits(a, n)

            b_bits = int_to_bits(b, n)

            statevector = emulator.run(
                _a_bit_array=a_bits, _b_bit_array=b_bits, ctrl_active=ctrl_active
            )
            states = Quest.extract_states_dict(statevector.results[0].entries)
            for state in states.values():
                assert state.get_single_state()[0] == 1


@pytest.mark.parametrize(
    ("n", "cases", "controlled_adder", "controlled_adder_dagger", "num_ancilla_fn"),
    [
        (n, cases, adder, dagger, ancillas)
        for adder, dagger, ancillas, widths in [
            (
                cntrl_adder_ripple_cuccaro_mod,
                cntrl_adder_ripple_cuccaro_mod_dagger,
                lambda n: 1,
                [2, 3, 4, 5, 6],
            ),
            (
                cntrl_adder_ripple_gidney_mod,
                cntrl_adder_ripple_gidney_mod_dagger,
                lambda n: n,
                # Stop at five bits: the six-bit inverse needs 25 simulator qubits.
                [2, 3, 4, 5],
            ),
        ]
        for n, cases in [
            (2, [(3, 1), (1, 2), (3, 3)]),
            (3, [(3, 4), (5, 2), (7, 7)]),
            (4, [(3, 7), (9, 4), (15, 15)]),
            (5, [(5, 12), (18, 6), (31, 31)]),
            (6, [(5, 32), (48, 6)]),
        ]
        if n in widths
    ],
)
def test_cntrl_addition_mod_dagger[n: nat](
    n: int,
    cases: list[tuple[int, int]],
    controlled_adder: GuppyFunctionDefinition[
        [qubit, array[qubit, n], array[qubit, n]],
        None,
    ],
    controlled_adder_dagger: GuppyFunctionDefinition[
        [qubit, array[qubit, n], array[qubit, n]],
        None,
    ],
    num_ancilla_fn: Callable[[int], int],
) -> None:
    """Test controlled modular-adder dagger inverts correctly."""
    n_qubits = 2 * n + 1 + 2 * num_ancilla_fn(n)

    @guppy
    @no_type_check
    def main(
        _a_bit_array: array[bool, n], _b_bit_array: array[bool, n], ctrl_active: bool
    ) -> None:
        a_reg = qarray(n)
        b_reg = qarray(n)
        apply_bitstring(a_reg, _a_bit_array)
        apply_bitstring(b_reg, _b_bit_array)
        carry_out = qubit()
        discard(carry_out)

        ctrl = qubit()
        if ctrl_active:
            x(ctrl)

        controlled_adder(ctrl, a_reg, b_reg)
        controlled_adder_dagger(ctrl, a_reg, b_reg)

        apply_bitstring(a_reg, _a_bit_array)
        apply_bitstring(b_reg, _b_bit_array)

        if ctrl_active:
            x(ctrl)

        state_output("ctrl", ctrl)
        state_output("a_reg", a_reg)
        state_output("b_reg", b_reg)

        discard(ctrl)
        discard_array(a_reg)
        discard_array(b_reg)

    # Compile once; reuse the program for every classical input below.
    emulator = main.emulator(n_qubits=n_qubits).with_seed(42)
    for ctrl_active in [False, True]:
        for a, b in cases:
            a_bits = int_to_bits(a, n)

            b_bits = int_to_bits(b, n)

            statevector = emulator.run(
                _a_bit_array=a_bits, _b_bit_array=b_bits, ctrl_active=ctrl_active
            )

            states = Quest.extract_states_dict(statevector.results[0].entries)

            for state in states.values():
                assert state.get_single_state()[0] == 1
