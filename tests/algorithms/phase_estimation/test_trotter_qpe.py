"""QPE tests using first-order Trotterized power oracles."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest
from guppylang.defs import GuppyFunctionDefinition
from hugr.qsystem.result import QsysResult
from pytest_lazy_fixtures import lf as lazy_fixture
import zixy.qubit.pauli as zqp
from selene_sim import Quest

from guppyalgos.algorithms.time_evolution.trotter import cntrl_trotter_first_order
from guppyalgos.utils import phase_distance_mod_2, trotterized_eigenphases
from guppyalgos.testing import assert_allclose_ignorephase
from tests.algorithms.phase_estimation.qpe_test_helpers import (
    dominant_trotter_measured_phase,
    exact_kickback_state,
    make_ry_state,
    make_trotter_qpe_diagnostic_program,
    make_trotter_qpe_program,
    make_trotter_power_oracle,
    phase_key,
    ry_state_probabilities,
)


def _single_shot_states(
    program: GuppyFunctionDefinition[[], None], n_qubits: int
) -> dict[str, Any]:
    """Run a diagnostic program once and return its state snapshots."""
    result = program.emulator(n_qubits=n_qubits).with_seed(5).with_shots(1).run()
    return Quest.extract_states_dict(QsysResult(result).results[0].entries)


def _single_shot_qpe_counts(
    program: GuppyFunctionDefinition[[], None], n_qubits: int
) -> dict[str, int]:
    """Run a measurement program once and return QPE bitstring counts."""
    result = program.emulator(n_qubits=n_qubits).with_seed(5).with_shots(1).run()
    return result.register_counts()["qpe_bitstring"]


def _state_component_probabilities(state: Any) -> list[float]:
    """Return probabilities from a state-vector distribution result."""
    return [
        measurable.probability for measurable in state.get_state_vector_distribution()
    ]


@pytest.mark.parametrize("n_ancilla", [3, 4])
@pytest.mark.parametrize("phi", [0.25, 0.75])
@pytest.mark.parametrize("eigenmix", [0, -0.33, 0.5])
def test_trotter_qpe_rz_toy_model_exact_numerics(
    n_ancilla: int, phi: float, eigenmix: float
) -> None:
    """Check the Trotter oracle path against the exact single-qubit Rz model.

    This is a bridge test: the Hamiltonian and timestep are chosen so one
    controlled Trotter step implements the same eigenphase kickback as the exact
    Rz oracle tests.
    """
    # For H = 0.5 * Z, choosing t = -4 phi makes the controlled Trotter step
    # implement the same single-eigenphase kickback used in the exact Rz tests.
    time_step = -4 * phi
    ham_op = zqp.RealTermSum.from_str("(0.5, Z0)")
    controlled_step = cntrl_trotter_first_order(ham_op, 1)
    ground_phase = phi % 2
    excited_phase = (-phi) % 2
    state_preparation = make_ry_state(float(eigenmix))
    power_oracle = make_trotter_power_oracle(controlled_step, time_step, 1)
    diagnostic_program = make_trotter_qpe_diagnostic_program(
        n_ancilla,
        1,
        state_preparation,
        power_oracle,
    )
    measurement_program = make_trotter_qpe_program(
        n_ancilla,
        1,
        state_preparation,
        power_oracle,
    )
    states = _single_shot_states(diagnostic_program, n_ancilla + 1)
    counts = _single_shot_qpe_counts(measurement_program, n_ancilla + 1)

    kickback_state_probabilities = _state_component_probabilities(
        states["kickback_register"]
    )
    post_qpe_probabilities = _state_component_probabilities(states["qpe_register"])
    expected_state = exact_kickback_state(ground_phase, n_ancilla)
    ground_key = phase_key(ground_phase, n_ancilla)
    excited_key = phase_key(excited_phase, n_ancilla)
    expected_component_probabilities = sorted(
        probability
        for probability in ry_state_probabilities(eigenmix)
        if not np.isclose(probability, 0.0)
    )
    expected_system_state = np.array(
        [
            np.cos(np.pi * eigenmix / 2),
            np.sin(np.pi * eigenmix / 2),
        ],
        dtype=np.complex128,
    )

    if len(expected_component_probabilities) == 1:
        # Pure eigenstate prep should produce the exact analytical phase ramp.
        assert_allclose_ignorephase(
            states["kickback_register"].get_single_state(), expected_state
        )
        # In the pure-eigenstate case, the controlled Trotter oracle should only
        # add phase and leave the system register in the same eigenstate.
        assert_allclose_ignorephase(
            states["system_register"].get_single_state(),
            expected_system_state,
        )
        # Exact dyadic phases should be measured deterministically.
        assert counts.get(ground_key, 0) == 1
    else:
        # Superposition prep should preserve branch weights before the inverse QFT.
        assert np.allclose(
            sorted(kickback_state_probabilities), expected_component_probabilities
        )
        # The final QPE register should carry the same branch weights.
        assert np.allclose(
            sorted(post_qpe_probabilities), expected_component_probabilities
        )
        # Measurement should only return one of the two exact eigenphase branches.
        assert set(counts.keys()).issubset({ground_key, excited_key})
        # The diagnostic measurement program is run with a single shot.
        assert sum(counts.values()) == 1


def _assert_trotter_qpe_resolves_trotterized_phase(
    n_ancilla: int,
    ham_op: zqp.RealTermSum,
    time_step: float,
    prepared_state: NDArray[np.complex128],
    state_preparation: Any,
    min_dominant_overlap: float,
) -> None:
    """Check that QPE resolves an eigenphase of the implemented Trotter step."""
    n_state_qubits = len(ham_op.qubits)
    controlled_step = cntrl_trotter_first_order(ham_op, n_state_qubits)
    power_oracle = make_trotter_power_oracle(
        controlled_step,
        time_step,
        n_state_qubits,
    )
    phase_resolution = 1 / (2**n_ancilla)

    trotter_phases, trotter_eigenvectors = trotterized_eigenphases(
        ham_op,
        time_step,
        little_endian=True,
    )
    trotter_overlaps = np.abs(trotter_eigenvectors.conj().T @ prepared_state) ** 2
    dominant_trotter_overlap = float(np.max(trotter_overlaps))
    dominant_measured_phase = dominant_trotter_measured_phase(
        n_ancilla,
        n_state_qubits,
        state_preparation,
        power_oracle,
        shots=50,
    )
    phase_distance_to_trotter = min(
        phase_distance_mod_2(dominant_measured_phase, phase) for phase in trotter_phases
    )

    if np.isclose(min_dominant_overlap, 1.0):
        # Exact eigenstate inputs should align with one Trotter eigenvector.
        assert np.isclose(dominant_trotter_overlap, 1.0)
    else:
        # Approximate state prep should still overlap enough with one eigenphase.
        assert dominant_trotter_overlap > min_dominant_overlap

    # The measured QPE peak should land within one ancilla bin of a Trotter phase.
    assert phase_distance_to_trotter <= phase_resolution


@pytest.mark.parametrize("n_ancilla", [3, 4])
def test_trotter_qpe_commuting_comp_basis_examples(
    n_ancilla: int,
    ham_commuting_comp_basis_fixture: tuple[zqp.RealTermSum, float],
) -> None:
    """Check QPE resolves Trotter phases for computational-basis eigenstates.

    These commuting fixtures are diagonal in the computational basis, so
    |0...0> is an exact eigenstate of the first-order Trotter step.
    """
    ham_op, time_step = ham_commuting_comp_basis_fixture
    prepared_state = np.zeros(2 ** len(ham_op.qubits), dtype=np.complex128)
    prepared_state[0] = 1.0

    _assert_trotter_qpe_resolves_trotterized_phase(
        n_ancilla,
        ham_op,
        time_step,
        prepared_state,
        make_ry_state(0),
        min_dominant_overlap=1.0,
    )


@pytest.mark.parametrize("n_ancilla", [3, 4])
@pytest.mark.parametrize(
    "ham_op_time_step",
    [
        lazy_fixture("ham_noncommuting_example_0"),
        lazy_fixture("ham_noncommuting_example_1"),
    ],
)
def test_trotter_qpe_noncommuting_one_qubit_examples(
    n_ancilla: int,
    ham_op_time_step: tuple[zqp.RealTermSum, float],
) -> None:
    """Check QPE resolves Trotter phases for real one-qubit ground states."""
    ham_op, time_step = ham_op_time_step
    # These fixtures are intentionally limited to single-qubit state prep.
    assert len(ham_op.qubits) == 1

    ham_mat = ham_op.to_sparse_matrix(True).toarray()
    energies, eigenvectors = np.linalg.eigh(ham_mat)
    prepared_state = np.real_if_close(
        np.asarray(eigenvectors[:, 0], dtype=np.complex128)
    )
    # A single Ry state-prep angle is valid only for real eigenvectors.
    assert not np.iscomplexobj(prepared_state)

    prepared_state = np.asarray(prepared_state, dtype=float)
    prepared_state /= np.linalg.norm(prepared_state)
    if prepared_state[0] < 0 or (
        np.isclose(prepared_state[0], 0.0) and prepared_state[1] < 0
    ):
        prepared_state = -prepared_state

    # The chosen state is the ground state of a nontrivial signed Hamiltonian.
    assert float(energies[0]) < 0
    state_prep_angle = float(2 * np.arctan2(prepared_state[1], prepared_state[0]))

    _assert_trotter_qpe_resolves_trotterized_phase(
        n_ancilla,
        ham_op,
        time_step,
        prepared_state.astype(np.complex128),
        make_ry_state(state_prep_angle),
        min_dominant_overlap=0.5,
    )


# TODO: Add an exact-state-preparation integration test for a small multi-qubit
# eigenstate, then use it to cover both a non-computational-basis commuting
# Hamiltonian and a higher-qubit noncommuting Hamiltonian.
# TODO: Add an H2 STO-3G Trotter-QPE integration test covering the Hartree-Fock
# initial state, dominant measured phase, and phase-to-energy conversion.
# TODO: Add a histogram-level check for a representative noncommuting example so
# we assert more than just the dominant measured phase when runtime allows.
