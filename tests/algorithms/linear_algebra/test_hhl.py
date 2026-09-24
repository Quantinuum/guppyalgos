"""Tests for the HHL algorithm."""

from __future__ import annotations
from typing import no_type_check

import numpy as np
import pytest
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard, discard_array, measure, qubit
from selene_sim import QuantumReplay, Quest

from guppyalgos.algorithms.linear_algebra import (
    eigenvalue_inversion,
    hhl,
)
from guppyalgos.algorithms.time_evolution.trotter import (
    cntrl_ham_sim_trotter,
    cntrl_trotter_first_order,
)
from guppyalgos.algorithms.time_evolution.trotter.trotter_sequence import (
    cntrl_trotter_from_sequence,
)
from guppyalgos.algorithms.linear_algebra.hhl_utils import eigenvalue_inversion_angles
from guppyalgos.algorithms.state_preparation import multiplexor_prep
from guppyalgos.utils import apply_bitstring, int_to_bits, qarray
from guppyalgos.primitives.measurement import discard_array_zero
from tests.helpers.test_helpers import assert_allclose_ignorephase, switch_endianness


def test_eigenvalue_inversion_angles_use_signed_clock_labels() -> None:
    """The inversion angles encode C divided by each inferred eigenvalue."""
    angles = eigenvalue_inversion_angles(
        n_qpe=3, rotation_scalar=0.25, simulation_time=np.pi / 2
    )

    expected = [
        0.0,
        1.0 / 3.0,
        2.0 / np.pi * np.arcsin(1.0 / 4.0),
        2.0 / np.pi * np.arcsin(1.0 / 6.0),
        -2.0 / np.pi * np.arcsin(1.0 / 8.0),
        -2.0 / np.pi * np.arcsin(1.0 / 6.0),
        -2.0 / np.pi * np.arcsin(1.0 / 4.0),
        -1.0 / 3.0,
    ]

    assert angles == pytest.approx(expected)


@pytest.mark.parametrize(
    (
        "ham_str",
        "n_input_qubits",
        "input_vector",
        "n_qpe",
        "simulation_time",
        "rotation_scalar",
    ),
    [
        (
            "(1.5, I0), (-0.5, Z0)",
            1,
            np.array([np.sqrt(3) / 2, 0.5]),
            3,
            -0.5,
            1.0,
        ),
        (
            "(0.5, I0), (1.5, Z0)",
            1,
            np.array([1.0, 1.0]) / np.sqrt(2),
            3,
            -0.5,
            1.0,
        ),
        (
            "(1.5, I0 I1), (-0.5, Z0 I1)",
            2,
            np.full(4, 0.5),
            3,
            -0.5,
            1.0,
        ),
        (
            "(1.5, I0 I1), (-0.5, Z0 Z1)",
            2,
            np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2),
            3,
            -0.5,
            1.0,
        ),
        (
            "(1.5, I0 I1 I2), (0.5, X0 Z1 Z2)",
            3,
            np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) / np.sqrt(2),
            3,
            -0.5,
            1.0,
        ),
        (
            "(0.5, I0 I1 I2), (1.5, Z0 Z1 Z2)",
            3,
            np.full(8, 1.0 / np.sqrt(8)),
            3,
            -0.5,
            1.0,
        ),
    ],
)
def test_hhl_rus(
    ham_str: str,
    n_input_qubits: int,
    input_vector: np.ndarray,
    n_qpe: int,
    simulation_time: float,
    rotation_scalar: float,
) -> None:
    """Test HHL across 1-, 2-, and 3-qubit linear systems with repeat-until-success."""
    ham_op = zqp.RealTermSum.from_str(ham_str)

    n_sim_qubits = n_qpe + n_input_qubits + 1
    controlled_trotter_step = cntrl_trotter_first_order(
        ham_op, n_state_qubits=n_input_qubits
    )
    ham_terms = list(ham_op.to_terms())
    inverse_trotter_step = cntrl_trotter_from_sequence(
        ham_terms,
        [(term_index, 1.0) for term_index in reversed(range(len(ham_terms)))],
        n_input_qubits,
    )
    forward_simulation = cntrl_ham_sim_trotter(
        controlled_trotter_step, 1, simulation_time, n_input_qubits
    )
    inverse_simulation = cntrl_ham_sim_trotter(
        inverse_trotter_step, 1, -simulation_time, n_input_qubits
    )

    @guppy
    @no_type_check
    def controlled_hamiltonian_simulation(
        control: qubit,
        state_register: array[qubit, n_input_qubits],
        power: int,
    ) -> None:
        if power >= 0:
            for _ in range(power):
                forward_simulation(control, state_register)
        else:
            for _ in range(-power):
                inverse_simulation(control, state_register)

    @guppy
    @no_type_check
    def eigenvalue_transform(clock_reg: array[qubit, n_qpe], ancilla: qubit) -> None:
        eigenvalue_inversion(
            clock_reg,
            ancilla,
            comptime(rotation_scalar),
            comptime(simulation_time),
        )

    prepare_b = multiplexor_prep(input_vector)

    @guppy
    @no_type_check
    def run_hhl_rus() -> None:
        while True:
            qs = qarray(n_input_qubits)
            prepare_b(qs)
            clock_reg = qarray(n_qpe)
            ancilla = qubit()
            hhl(
                qs,
                clock_reg,
                ancilla,
                controlled_hamiltonian_simulation,
                eigenvalue_transform,
            )
            success = measure(ancilla).read()
            if success:
                state_output("solution", qs)
                discard_array(clock_reg)
                discard_array(qs)
                break
            discard_array(clock_reg)
            discard_array(qs)

    a_mat = ham_op.to_sparse_matrix(False).toarray()
    expected_x = np.linalg.solve(a_mat, input_vector)
    expected_x /= np.linalg.norm(expected_x)

    n_shots = 20
    desired_measurements = [[False] * n + [True] for n in range(n_shots)]
    replay_simulator = QuantumReplay(
        simulator=Quest(), measurements=desired_measurements
    )
    replay_result = (
        run_hhl_rus.emulator(n_qubits=n_sim_qubits)
        .with_simulator(replay_simulator)
        .with_shots(n_shots)
        .run()
    )

    for shot_result in replay_result.results:
        states = Quest.extract_states_dict(shot_result)
        actual_state = switch_endianness(
            states["solution"].get_state_vector_distribution()[0].state
        )
        assert_allclose_ignorephase(actual_state, expected_x, threshold=1e-5)


@pytest.mark.parametrize(
    (
        "clock_reg_size",
        "scaling_factor",
        "simulation_time",
        "clock_reg_state",
        "n_ancillas",
    ),
    [
        (2, 1.0, np.pi / 2, 0.5, 0),
        (3, 0.1, -np.pi, 0.6, 0),
    ],
)
def test_eigenvalue_inversion[n_clock: nat](
    clock_reg_size: int,
    scaling_factor: float,
    simulation_time: float,
    clock_reg_state: float,
    n_ancillas: int,
) -> None:
    """Test that the eigenvalue inversion function has the expected behavior.

    For an n-qubit clock basis state |k⟩, interpret k as a signed integer label and
    infer λ = 2πk / (t 2^n). The inversion should perform
    |k⟩|0⟩ → |k⟩(√(1 - |C/λ|^2)|0⟩ + (C/λ)|1⟩).

    """
    clock_reg_state_int = int(clock_reg_state * (2**clock_reg_size))
    clock_reg_state_bits = int_to_bits(clock_reg_state_int, clock_reg_size)
    signed_clock_label = (
        clock_reg_state_int
        if clock_reg_state_int < 2 ** (clock_reg_size - 1)
        else clock_reg_state_int - 2**clock_reg_size
    )
    eigenvalue = (
        2.0 * np.pi * signed_clock_label / (simulation_time * 2**clock_reg_size)
    )
    expected_amplitude = scaling_factor / eigenvalue
    expected_state = np.array(
        [np.sqrt(1.0 - expected_amplitude**2), expected_amplitude]
    )

    @guppy
    @no_type_check
    def main() -> None:
        clock_reg = qarray(clock_reg_size)
        apply_bitstring(clock_reg, clock_reg_state_bits)

        ancilla = qubit()
        eigenvalue_inversion(
            clock_reg,
            ancilla,
            comptime(scaling_factor),
            comptime(simulation_time),
        )

        apply_bitstring(clock_reg, clock_reg_state_bits)
        discard_array_zero(clock_reg)

        state_output("ancilla", ancilla)
        discard(ancilla)

    sim_result = main.emulator(n_qubits=clock_reg_size + n_ancillas + 1).run()
    states = Quest.extract_states_dict(sim_result.results[0].entries)
    actual_state = states["ancilla"].get_state_vector_distribution()[0].state

    assert_allclose_ignorephase(actual_state, expected_state)
