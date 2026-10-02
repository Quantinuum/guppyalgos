"""CX ladder tests."""

from collections.abc import Callable
from itertools import product
from typing import Any, no_type_check

import numpy as np
import pytest
from guppylang import comptime, guppy
from guppylang.std.builtins import array, control, dagger
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, h, qubit
from selene_sim import Quest

from guppyalgos.primitives.subroutines.ladders.cx_ladder import (
    CXLadderLinear,
    CXLadderLog,
    _cx_ladder_apply_from_inds,
    ladder_inds_from_ascending,
    _lin_cx_ladder_indices,
    log_cx_ladder_indices,
)
from guppyalgos.primitives.subroutines.ladders.ladder import LadderIndexing
from guppyalgos.primitives.state_preparation.uniform import uniform_state
from guppyalgos.utils import qarray

from guppyalgos.testing import (
    align_phase,
    assert_allclose_ignorephase,
    assert_cntrl_unitary,
    get_unitary,
)


@pytest.mark.parametrize(
    ("ladder_indices_fnc", "expected_indices"),
    [
        (
            _lin_cx_ladder_indices,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
        ),
        (
            lambda n: ladder_inds_from_ascending(
                _lin_cx_ladder_indices(n), LadderIndexing.ASCENDING_DAGGER
            ),
            [(5, 6), (4, 5), (3, 4), (2, 3), (1, 2), (0, 1)],
        ),
        (
            lambda n: ladder_inds_from_ascending(
                _lin_cx_ladder_indices(n), LadderIndexing.DESCENDING
            ),
            [(6, 5), (5, 4), (4, 3), (3, 2), (2, 1), (1, 0)],
        ),
        (
            lambda n: ladder_inds_from_ascending(
                _lin_cx_ladder_indices(n), LadderIndexing.DESCENDING_DAGGER
            ),
            [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5)],
        ),
    ],
)
def test_linear_cx_ladder_indices(
    ladder_indices_fnc: Callable[[int], list[tuple[int, int]]],
    expected_indices: list[tuple[int, int]],
) -> None:
    """Test the linear CX ladder indices generation."""
    n_qubits = 7
    assert ladder_indices_fnc(n_qubits) == expected_indices


@pytest.mark.parametrize(
    ("ladder_indices_fnc", "expected_indices"),
    [
        (
            log_cx_ladder_indices,
            [(0, 1), (2, 3), (4, 5), (1, 3), (3, 5), (3, 4), (1, 2), (5, 6)],
        ),
        (
            lambda n: ladder_inds_from_ascending(
                log_cx_ladder_indices(n), LadderIndexing.ASCENDING_DAGGER
            ),
            [(5, 6), (1, 2), (3, 4), (3, 5), (1, 3), (4, 5), (2, 3), (0, 1)],
        ),
        (
            lambda n: ladder_inds_from_ascending(
                log_cx_ladder_indices(n), LadderIndexing.DESCENDING
            ),
            [(6, 5), (2, 1), (4, 3), (5, 3), (3, 1), (5, 4), (3, 2), (1, 0)],
        ),
        (
            lambda n: ladder_inds_from_ascending(
                log_cx_ladder_indices(n), LadderIndexing.DESCENDING_DAGGER
            ),
            [(1, 0), (3, 2), (5, 4), (3, 1), (5, 3), (4, 3), (2, 1), (6, 5)],
        ),
    ],
)
def test_log_cx_ladder_indices(
    ladder_indices_fnc: Callable[[int], list[tuple[int, int]]],
    expected_indices: list[tuple[int, int]],
) -> None:
    """Test the log CX ladder indices generation."""
    n_qubits = 7
    assert ladder_indices_fnc(n_qubits) == expected_indices


@pytest.mark.parametrize(
    ("n_qubits"),
    list(range(1, 11)),
)
def test_linear_and_log_depth_unitary_equivalence(n_qubits: int) -> None:
    """Test that linear and log CX ladder indices produce the same unitaries."""

    @guppy
    def lin(qs: array[qubit, n_qubits]) -> None:
        CXLadderLinear().ascending(qs)

    @guppy
    def log(qs: array[qubit, n_qubits]) -> None:
        CXLadderLog().ascending(qs)

    assert_allclose_ignorephase(get_unitary(lin, n_qubits), get_unitary(log, n_qubits))


def test_cx_ladder_inverse() -> None:
    """Test the custom inverse implementation of the CX ladder."""
    n_qubits = 4
    gate_indices = log_cx_ladder_indices(n_qubits)

    @guppy
    @no_type_check
    def ladder(qs: array[qubit, n_qubits]) -> None:
        _cx_ladder_apply_from_inds(qs, comptime(gate_indices))

    @guppy
    @no_type_check
    def inverse_ladder(qs: array[qubit, n_qubits]) -> None:
        with dagger:
            _cx_ladder_apply_from_inds(qs, comptime(gate_indices))

    ladder_unitary = get_unitary(ladder, n_qubits)
    inverse_unitary = get_unitary(inverse_ladder, n_qubits)

    assert_allclose_ignorephase(inverse_unitary, ladder_unitary.conj().T)


def test_controlled_cx_ladder() -> None:
    """Test the custom controlled implementation of the CX ladder."""
    n_qubits = 4
    gate_indices = log_cx_ladder_indices(n_qubits)

    @guppy
    @no_type_check
    def ladder(qs: array[qubit, n_qubits]) -> None:
        _cx_ladder_apply_from_inds(qs, comptime(gate_indices))

    @guppy
    @no_type_check
    def controlled_ladder(ctrl: array[qubit, 1], qs: array[qubit, n_qubits]) -> None:
        h(ctrl[0])
        with control(ctrl):
            _cx_ladder_apply_from_inds(qs, comptime(gate_indices))
        h(ctrl[0])

    assert_cntrl_unitary(
        controlled_ladder,
        align_phase(get_unitary(ladder, n_qubits)),
        n_qubits,
        {},
        control_name="ctrl",
    )


@pytest.mark.parametrize(
    ("n_qubits", "ladder"),
    list(
        product(
            [5, 8, 11],
            [CXLadderLinear, CXLadderLog],
        )
    ),
)
def test_get_cx_ladder_impl_basic_sanity(n_qubits: int, ladder: Any) -> None:
    """Basic sanity check for the CX ladder implementation.

    Any ladder implementation must be an identity on all-zero state.
    """

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

    res = main.emulator(n_qubits).run()
    states = Quest.extract_states_dict(res.results[0].entries)
    for state in states.values():
        expected_state_all0 = np.array([1] + [0] * (2**n_qubits - 1))
        assert_allclose_ignorephase(state.get_single_state(), expected_state_all0)


@pytest.mark.parametrize(
    ("n_qubits", "ladder"),
    list(
        product(
            [5, 8, 11],
            [CXLadderLinear, CXLadderLog],
        )
    ),
)
def test_get_cx_ladder_impl(n_qubits: int, ladder: Any) -> None:
    """Test the CX ladder implementation.

    Any ladder implementation would be an identity on the uniform state, so we can
    use the uniform state preparation to verify the correctness of the CX ladder.
    """
    uniform = uniform_state(2**n_qubits)

    @guppy
    @no_type_check
    def main() -> None:
        qs = qarray(n_qubits)
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

    res = main.emulator(n_qubits).run()
    states = Quest.extract_states_dict(res.results[0].entries)
    for state in states.values():
        uniform_state_expected = np.array([1 / (2 ** (n_qubits / 2))] * (2**n_qubits))
        assert_allclose_ignorephase(state.get_single_state(), uniform_state_expected)
