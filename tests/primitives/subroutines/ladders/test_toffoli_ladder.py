"""Toffoli ladder tests."""

import random
from collections.abc import Callable
from itertools import product
from typing import Any, no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.debug import state_output
from guppylang.std.angles import angle
from guppylang.std.quantum import (
    collect_measurements,
    discard_array,
    measure_array,
    qubit,
    ry,
)
from guppylang.std.builtins import comptime
from guppyalgos.primitives.measurement.utils import discard_array_zero
from guppyalgos.utils import apply_bitstring, int_to_bits, qarray
from selene_sim import Quest

from guppyalgos.primitives.subroutines.ladders.toffoli_ladder import (
    ToffoliLadderLinear,
    ToffoliLadderLog,
    _ladder_inds_from_ascending,
    _lin_toffoli_ladder_indices,
    log_toffoli_ladder_num_ancilla,
)
from guppyalgos.primitives.subroutines.ladders.ladder import LadderIndexing
from guppyalgos.primitives.state_preparation.uniform import uniform_state

from guppyalgos.testing import assert_allclose_ignorephase


@pytest.mark.parametrize(
    ("ladder_indices_fnc", "expected_indices"),
    [
        (
            _lin_toffoli_ladder_indices,
            [(0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 8)],
        ),
        (
            lambda n: _ladder_inds_from_ascending(
                _lin_toffoli_ladder_indices(n), LadderIndexing.ASCENDING_DAGGER
            ),
            [(6, 7, 8), (4, 5, 6), (2, 3, 4), (0, 1, 2)],
        ),
        (
            lambda n: _ladder_inds_from_ascending(
                _lin_toffoli_ladder_indices(n), LadderIndexing.DESCENDING
            ),
            [(8, 7, 6), (6, 5, 4), (4, 3, 2), (2, 1, 0)],
        ),
        (
            lambda n: _ladder_inds_from_ascending(
                _lin_toffoli_ladder_indices(n), LadderIndexing.DESCENDING_DAGGER
            ),
            [(2, 1, 0), (4, 3, 2), (6, 5, 4), (8, 7, 6)],
        ),
    ],
)
def test_linear_toffoli_ladder_indices(
    ladder_indices_fnc: Callable[[int], list[tuple[int, int, int]]],
    expected_indices: list[tuple[int, int, int]],
) -> None:
    """Test the linear Toffoli ladder indices generation."""
    n_qubits = 9
    assert ladder_indices_fnc(n_qubits) == expected_indices


@pytest.mark.parametrize(("n_qubits"), list(range(5, 11, 2)))
def test_linear_and_log_depth_basis_equivalence(n_qubits: int) -> None:
    """Test that linear and log Toffoli ladders act identically."""

    def run_ladder(ladder: Any):
        @guppy
        @no_type_check
        def run(bits: array[bool, n_qubits]) -> None:
            qs = qarray(n_qubits)
            apply_bitstring(qs, bits)

            for i in range(n_qubits):
                ry(qs[i], angle(0.37))

            ladder().ascending(qs)
            state_output("res_asc", qs)

            ladder().descending(qs)
            state_output("res_desc", qs)

            ladder().ascending_dagger(qs)
            state_output("res_ascdag", qs)

            ladder().descending_dagger(qs)
            state_output("res_descdag", qs)

            discard_array(qs)

        return run

    input_values = [2, 5]

    # Compile each implementation once, then reuse it for both classical inputs.
    linear = run_ladder(ToffoliLadderLinear).emulator(n_qubits=n_qubits).with_seed(42)
    log = (
        run_ladder(ToffoliLadderLog)
        .emulator(n_qubits=n_qubits + log_toffoli_ladder_num_ancilla(n_qubits))
        .with_seed(42)
    )
    for input_value in input_values:
        bits = int_to_bits(input_value, n_qubits)
        linear_result = linear.run(bits=bits)
        log_result = log.run(bits=bits)

        linear_states = Quest.extract_states_dict(linear_result.results[0].entries)
        log_states = Quest.extract_states_dict(log_result.results[0].entries)

        assert linear_states.keys() == log_states.keys()

        for key in linear_states:
            linear_state = linear_states[key].get_single_state()
            log_state = log_states[key].get_single_state()
            np.testing.assert_allclose(
                linear_state,
                log_state,
                err_msg=f"Mismatch for input {input_value:0{n_qubits}b}",
            )


@pytest.mark.parametrize(
    ("n_qubits", "ladder", "n_anc_func"),
    [
        (n_qubits, ladder, n_anc_func)
        for n_qubits, (ladder, n_anc_func) in product(
            [5, 9, 11],
            [
                (ToffoliLadderLinear, lambda _: 0),
                (ToffoliLadderLog, log_toffoli_ladder_num_ancilla),
            ],
        )
    ],
)
def test_get_ccx_ladder_impl_basic_sanity(
    n_qubits: int,
    ladder: Any,
    n_anc_func: Callable[[int], int],
) -> None:
    """Basic sanity check for the Toffoli ladder implementation.

    Any ladder implementation must be an identity on all-zero state.
    """
    n_anc = n_anc_func(n_qubits)

    @guppy
    @no_type_check
    def main() -> None:
        qs = qarray(n_qubits)

        ladder().ascending(qs)
        state_output("res_asc", qs)

        ladder().descending(qs)
        state_output("res_desc", qs)

        ladder().ascending_dagger(qs)
        state_output("res_ascdag", qs)

        ladder().descending_dagger(qs)
        state_output("res_descdag", qs)

        discard_array(qs)

    res = main.emulator(n_qubits + n_anc).run()
    states = Quest.extract_states_dict(res.results[0].entries)

    for state in states.values():
        expected_state_all0 = np.array([1] + [0] * (2**n_qubits - 1))
        assert_allclose_ignorephase(state.get_single_state(), expected_state_all0)


@pytest.mark.parametrize(
    ("n_qubits", "ladder", "n_anc_func"),
    [
        (n_qubits, ladder, n_anc_func)
        for n_qubits, (ladder, n_anc_func) in product(
            [5, 9, 11],
            [
                (ToffoliLadderLinear, lambda _: 0),
                (ToffoliLadderLog, log_toffoli_ladder_num_ancilla),
            ],
        )
    ],
)
def test_get_toffoli_ladder_impl(
    n_qubits: int,
    ladder: Any,
    n_anc_func: Callable[[int], int],
) -> None:
    """Test the Toffoli ladder implementation.

    Any ladder implementation would be an identity on the uniform state, so we can
    use the uniform state preparation to verify the correctness of the Toffoli ladder.
    """
    n_anc = n_anc_func(n_qubits)
    uniform = uniform_state(2**n_qubits)

    @guppy
    @no_type_check
    def main() -> None:
        qs = qarray(comptime(n_qubits))
        uniform(qs)

        ladder().ascending(qs)
        state_output("res_asc", qs)

        ladder().descending(qs)
        state_output("res_desc", qs)

        ladder().ascending_dagger(qs)
        state_output("res_ascdag", qs)

        ladder().descending_dagger(qs)
        state_output("res_descdag", qs)

        discard_array(qs)

    res = main.emulator(n_qubits + n_anc).run()
    states = Quest.extract_states_dict(res.results[0].entries)

    for state in states.values():
        uniform_state_expected = np.array([1 / (2 ** (n_qubits / 2))] * (2**n_qubits))
        assert_allclose_ignorephase(state.get_single_state(), uniform_state_expected)


@pytest.mark.parametrize(
    ("n_qubits_per_ladder", "n_ladders", "ladder", "n_anc_func"),
    [
        (n_qubits_per_ladder, n_ladders, ladder, n_anc_func)
        for n_qubits_per_ladder, n_ladders, (ladder, n_anc_func) in product(
            [5],
            [2],
            [
                (ToffoliLadderLinear, lambda _: 0),
                (ToffoliLadderLog, log_toffoli_ladder_num_ancilla),
            ],
        )
    ],
)
def test_concat_ladders(
    n_qubits_per_ladder: int,
    n_ladders: int,
    ladder: Any,
    n_anc_func: Callable[[int], int],
) -> None:
    """Test the concatenation of Toffoli ladders.

    Any ladder implementation would be an identity on the uniform state, so we can
    use the uniform state preparation to verify the correctness of the CCX ladder.
    """
    n_anc_per_ladder = n_anc_func(n_qubits_per_ladder)

    n_qubits = n_ladders * n_qubits_per_ladder - (n_ladders - 1)
    n_concat_anc = n_ladders * n_anc_per_ladder
    n_full_anc = n_anc_func(n_qubits)

    uniform = uniform_state(2**n_qubits)

    @guppy.comptime
    def apply_ladders(qs: array[qubit, comptime(n_qubits)]) -> None:
        step = n_qubits_per_ladder - 1

        for ldx in range(n_ladders):
            start = ldx * step
            end = start + n_qubits_per_ladder

            ladder().ascending(qs[start:end])

    @guppy
    def apply_concat_ladders() -> None:
        qs = qarray(n_qubits)
        uniform(qs)

        apply_ladders(qs)

        state_output("res_asc_concat", qs)

        discard_array(qs)

    @guppy
    def apply_full_ladder() -> None:
        qs = qarray(comptime(n_qubits))
        uniform(qs)

        ladder().ascending(qs)
        state_output("res_asc_full", qs)

        discard_array(qs)

    res = apply_concat_ladders.emulator(n_qubits + n_concat_anc).run()
    states = Quest.extract_states_dict(res.results[0].entries)
    concat_state = states["res_asc_concat"].get_single_state()

    res = apply_full_ladder.emulator(n_qubits + n_full_anc).run()
    states = Quest.extract_states_dict(res.results[0].entries)
    full_state = states["res_asc_full"].get_single_state()

    assert_allclose_ignorephase(concat_state, full_state)


@pytest.mark.parametrize("n_qubits", [5, 9, 11])
def test_log_depth_ladder_with_cca(n_qubits: int) -> None:
    """Test the log depth ladder on caller-provided ancillae."""
    n_anc = log_toffoli_ladder_num_ancilla(n_qubits)

    @guppy
    @no_type_check
    def main(bits: array[bool, n_qubits], orientation: int) -> None:
        qs = qarray(n_qubits)
        anc = qarray(n_anc)
        apply_bitstring(qs, bits)
        ladder = ToffoliLadderLog()
        if orientation == 0:
            ladder.ascending_with_cca(qs, anc)
        elif orientation == 1:
            ladder.descending_with_cca(qs, anc)
        elif orientation == 2:
            ladder.ascending_dagger_with_cca(qs, anc)
        else:
            ladder.descending_dagger_with_cca(qs, anc)
        discard_array_zero(anc)
        output("qs", collect_measurements(measure_array(qs)))

    emulator = main.emulator(n_qubits=n_qubits + n_anc).with_seed(42).with_shots(1)

    rng = random.Random(n_qubits)
    values = [2**n_qubits - 1, *(rng.randrange(2**n_qubits) for _ in range(5))]

    for indexing in LadderIndexing:
        gates = _ladder_inds_from_ascending(
            _lin_toffoli_ladder_indices(n_qubits), indexing
        )
        for value in values:
            bits = int_to_bits(value, n_qubits)
            expected = list(bits)
            for i, j, k in gates:
                expected[k] ^= expected[i] and expected[j]

            shot = emulator.run(bits=bits, orientation=indexing.value)
            assert shot.collated_shots()[0]["qs"][0] == expected, (indexing, value)
