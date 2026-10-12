"""Exact QPE tests using controlled Rz as the toy oracle."""

import numpy as np
import pytest
from hugr.qsystem.result import QsysResult
from selene_sim import Quest

from guppyalgos.utils import binary_fraction, phase_distance_mod_2
from guppyalgos.testing import assert_allclose_ignorephase
from tests.algorithms.phase_estimation.qpe_test_helpers import (
    exact_kickback_density_matrix,
    exact_kickback_state,
    is_exactly_expressible,
    make_inverse_rz_power_oracle,
    make_ry_state,
    make_rz_power_oracle,
    make_simple_qpe_kickback_program,
    make_simple_qpe_program,
    make_simple_qpe_reversibility_program,
    numpy_uniform,
    ry_state_probabilities,
)


@pytest.mark.parametrize("n_ancilla", [1, 3, 4])
@pytest.mark.parametrize("phi", [0.25, -0.75, 0.33])
@pytest.mark.parametrize("eigenmix", [0, -0.33, 0.5])
def test_power_oracle_qpe(n_ancilla: int, phi: float, eigenmix: float) -> None:
    """Test QPE with an Rz(phi) Hamiltonian.

    A CRz gate encodes binary powers of phi. The system register is prepared in
    a superposition of the computational-basis eigenstates |0> and |1> using
    ``eigenmix``. Dyadic phases with respect to the ancilla count are exactly
    expressible.
    """
    state_preparation = make_ry_state(float(eigenmix))
    power_oracle = make_rz_power_oracle(phi)
    abstract_canonical_qpe = make_simple_qpe_program(
        n_ancilla, state_preparation, power_oracle
    )

    n_shots = 1

    sim_result = (
        abstract_canonical_qpe.emulator(n_qubits=n_ancilla + 1)
        .with_seed(5)
        .with_shots(n_shots)
        .run()
    )

    states = Quest.extract_states_dict(QsysResult(sim_result).results[0].entries)
    raw_bitstring = sim_result.results[0].as_dict()["qpe_bitstring"]
    assert isinstance(raw_bitstring, list)
    bitstring = tuple(bool(bit) for bit in raw_bitstring)
    measured_phase = binary_fraction(bitstring)
    ground_prob, excited_prob = ry_state_probabilities(eigenmix)

    if is_exactly_expressible(phi, n_ancilla):
        expected_phases = []
        if not np.isclose(ground_prob, 0.0):
            expected_phases.append(float(np.mod(phi, 2)))
        if not np.isclose(excited_prob, 0.0):
            expected_phases.append(float(np.mod(-phi, 2)))
        assert any(
            np.isclose(phase_distance_mod_2(measured_phase, expected_phase), 0.0)
            for expected_phase in expected_phases
        )

    uniform_state_result = (
        states["phase_uniform_check"].get_state_vector_distribution()[0].state
    )
    kickback_state_probabilities = [
        measurable.probability
        for measurable in states["kickback_register"].get_state_vector_distribution()
    ]
    post_qpe_probabilities = [
        measurable.probability
        for measurable in states["qpe_register"].get_state_vector_distribution()
    ]

    assert_allclose_ignorephase(uniform_state_result, numpy_uniform(n_ancilla))

    assert 0 < len(kickback_state_probabilities) < 3
    assert 0 < len(post_qpe_probabilities) < 3

    if is_exactly_expressible(phi, n_ancilla):
        # For exactly representable phases, QPE preserves the system-component
        # weights exactly. Compare the nonzero probabilities as an unordered set
        # because the simulator does not guarantee the returned component order.
        expected_component_probabilities = sorted(
            probability
            for probability in ry_state_probabilities(eigenmix)
            if not np.isclose(probability, 0.0)
        )
        assert np.allclose(
            sorted(kickback_state_probabilities), expected_component_probabilities
        )
        assert np.allclose(
            sorted(post_qpe_probabilities), expected_component_probabilities
        )


@pytest.mark.parametrize("n_ancilla", [3])
@pytest.mark.parametrize("phi", [0.25, -0.75])
@pytest.mark.parametrize("eigenmix", [0, -0.33])
def test_reversibility_unitarity(n_ancilla: int, phi: float, eigenmix: float) -> None:
    """Check that QPE followed by inverse QPE restores both registers."""
    state_preparation = make_ry_state(float(eigenmix))
    power_oracle = make_rz_power_oracle(phi)
    inverse_power_oracle = make_inverse_rz_power_oracle(phi)
    abstract_canonical_qpe = make_simple_qpe_reversibility_program(
        n_ancilla,
        state_preparation,
        power_oracle,
        inverse_power_oracle,
    )

    n_shots = 1

    sim_result = (
        abstract_canonical_qpe.emulator(n_qubits=n_ancilla + 1)
        .with_seed(5)
        .with_shots(n_shots)
        .run()
    )

    states = Quest.extract_states_dict(QsysResult(sim_result).results[0].entries)

    uniform_state_result = (
        states["phase_uniform_check"].get_state_vector_distribution()[0].state
    )
    uniform_state_2_result = (
        states["phase_uniform_check_2"].get_state_vector_distribution()[0].state
    )
    assert np.allclose(uniform_state_result, uniform_state_2_result)

    unitary_regs_result = (
        states["unitary_regs_check"].get_state_vector_distribution()[0].state
    )
    unitary_regs_result_2 = (
        states["unitary_regs_check_2"].get_state_vector_distribution()[0].state
    )
    assert np.allclose(unitary_regs_result, unitary_regs_result_2)


@pytest.mark.parametrize("n_ancilla", [1, 3, 4])
@pytest.mark.parametrize("phi", [0.33, 0.2, -0.4])
@pytest.mark.parametrize("eigenmix", [0, -0.33, 0.5])
def test_kickback_register_matches_exact_numerics(
    n_ancilla: int, phi: float, eigenmix: float
) -> None:
    """Check the exact pre-IQFT kickback state, including superposition cases."""
    state_preparation = make_ry_state(float(eigenmix))
    power_oracle = make_rz_power_oracle(phi)
    abstract_kickback_state = make_simple_qpe_kickback_program(
        n_ancilla, state_preparation, power_oracle
    )

    sim_result = (
        abstract_kickback_state.emulator(n_qubits=n_ancilla + 1)
        .with_seed(5)
        .with_shots(1)
        .run()
    )

    states = Quest.extract_states_dict(QsysResult(sim_result).results[0].entries)

    kickback_register = states["kickback_register"]
    # This checks that tracing out the system register leaves the phase
    # register in the correct mixed density matrix for a superposition input.
    assert np.allclose(
        kickback_register.get_density_matrix(),
        exact_kickback_density_matrix(phi, n_ancilla, eigenmix),
    )

    ground_prob, excited_prob = ry_state_probabilities(eigenmix)
    if np.isclose(ground_prob, 1.0):
        assert_allclose_ignorephase(
            kickback_register.get_single_state(), exact_kickback_state(phi, n_ancilla)
        )
    elif np.isclose(excited_prob, 1.0):
        assert_allclose_ignorephase(
            kickback_register.get_single_state(), exact_kickback_state(-phi, n_ancilla)
        )
