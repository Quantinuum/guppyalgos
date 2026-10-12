"""Depth-1 Pauli phasor tests against the standard Pauli exponential."""

from __future__ import annotations

import itertools
from typing import no_type_check

import pytest
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import comptime
from guppylang.std.debug import state_result
from guppylang.std.quantum import discard_array, h
from selene_sim import QuantumReplay, Quest

import zixy.qubit.pauli as zqp

from guppyalgos.primitives.pauli.pauli_exp.pauli_exp_depth1 import pauli_exp_depth1
from guppyalgos.utils import qarray
from guppyalgos.utils.guppy.gates import transversal

from guppyalgos.testing import assert_allclose_ignorephase
from tests.primitives.pauli.pauli_exp.pauli_exp_helpers import pauli_exp_matrix

import numpy as np


REPRESENTATIVE_STRINGS = [
    ("I0 Z1", 2),
    ("X0 Y1", 2),
    ("Z0 X1 Y2", 3),
    ("I0 X1 Y2", 3),
]


def _all_measurement_branches(n_measures: int) -> list[list[bool]]:
    return [list(bits) for bits in itertools.product([True, False], repeat=n_measures)]


@pytest.mark.parametrize(("p_str", "n_state_qubits"), REPRESENTATIVE_STRINGS)
def test_pauli_phasor_depth1_matches_pauli_exp_for_all_measurement_branches(
    p_str: str,
    n_state_qubits: int,
) -> None:
    """Depth-1 phasor should match ``pauli_exp_matrix`` on every replayed branch."""
    pauli_string = zqp.String.from_str(p_str, n_state_qubits)
    theta = 0.3

    depth1_gadget = pauli_exp_depth1(pauli_string, n_state_qubits)

    ref_matrix = pauli_exp_matrix(pauli_string, n_state_qubits, theta)
    state0 = np.zeros(2**n_state_qubits)
    state0[0] = 1
    ref_state = ref_matrix @ state0

    @guppy
    @no_type_check
    def depth1_main() -> None:
        qreg = qarray(comptime(n_state_qubits))
        depth1_gadget(qreg, angle(comptime(theta)))
        state_result("result_state", qreg)
        discard_array(qreg)

    # n_active = len(pauli_string.get_dict())
    replay_measurements = _all_measurement_branches(n_state_qubits)

    replay_simulator = QuantumReplay(
        simulator=Quest(random_seed=17),
        resume_with_measurement=True,
        measurements=replay_measurements,
    )

    replay_result = (
        depth1_main.emulator(n_state_qubits + n_state_qubits)
        .with_simulator(replay_simulator)
        .with_shots(len(replay_measurements))
        .run()
    )

    for shot_result in replay_result.results:
        branch_state = Quest.extract_states_dict(shot_result)[
            "result_state"
        ].get_single_state()
        assert_allclose_ignorephase(branch_state, ref_state)


@pytest.mark.parametrize(("p_str", "n_state_qubits"), REPRESENTATIVE_STRINGS)
def test_pauli_phasor_depth1_matches_pauli_exp_for_all_measurement_branches2(
    p_str: str,
    n_state_qubits: int,
) -> None:
    """Depth-1 phasor should match ``pauli_exp_matrix`` on every replayed branch."""
    pauli_string = zqp.String.from_str(p_str, n_state_qubits)
    theta = 0.3

    depth1_gadget = pauli_exp_depth1(pauli_string, n_state_qubits)

    ref_matrix = pauli_exp_matrix(pauli_string, n_state_qubits, theta)
    state0 = (1 / np.sqrt(2**n_state_qubits)) * np.ones(2**n_state_qubits)
    ref_state = ref_matrix @ state0

    @guppy
    @no_type_check
    def depth1_main() -> None:
        qreg = qarray(comptime(n_state_qubits))
        transversal(h, qreg)
        depth1_gadget(qreg, angle(comptime(theta)))
        state_result("result_state", qreg)
        discard_array(qreg)

    # n_active = len(pauli_string.get_dict())
    replay_measurements = _all_measurement_branches(n_state_qubits)

    replay_simulator = QuantumReplay(
        simulator=Quest(random_seed=17),
        resume_with_measurement=True,
        measurements=replay_measurements,
    )

    replay_result = (
        depth1_main.emulator(n_state_qubits + n_state_qubits)
        .with_simulator(replay_simulator)
        .with_shots(len(replay_measurements))
        .run()
    )

    for shot_result in replay_result.results:
        branch_state = Quest.extract_states_dict(shot_result)[
            "result_state"
        ].get_single_state()
        assert_allclose_ignorephase(branch_state, ref_state)
