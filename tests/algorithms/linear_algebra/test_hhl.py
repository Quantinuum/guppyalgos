"""Tests for the HHL algorithm."""

from __future__ import annotations
from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.debug import state_output
from guppylang.std.quantum import crz, discard, discard_array, qubit
from selene_sim import Quest

from guppyalgos.algorithms.linear_algebra import (
    eigenvalue_inversion,
    hhl,
)
from guppyalgos.algorithms.linear_algebra.hhl_utils import eigenvalue_inversion_angles
from guppyalgos.algorithms.state_preparation import multiplexor_prep
from guppyalgos.primitives.rotations import (
    RotationAxis,
    RotationAxisX,
    RotationAxisY,
    RotationAxisZ,
)
from guppyalgos.testing import project_state_onto_bitstring
from guppyalgos.utils import apply_bitstring, int_to_bits, qarray
from guppyalgos.primitives.measurement import discard_array_zero
from tests.helpers.test_helpers import assert_allclose_ignorephase


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
        "rotation_axis",
        "coefficient",
        "input_vector",
        "n_qpe",
        "simulation_time",
        "rotation_scalar",
    ),
    [
        (
            RotationAxisZ,
            1.0,
            np.array([np.sqrt(3) / 2, 0.5]),
            3,
            -0.5,
            1.0,
        ),
        (
            RotationAxisX,
            1.5,
            np.array([1.0, 1.0]) / np.sqrt(2),
            3,
            -1.0 / 3.0,
            0.75,
        ),
        (
            RotationAxisX,
            1.0,
            np.array([np.sqrt(3) / 2, 0.5]),
            3,
            -0.5,
            0.75,
        ),
        (
            RotationAxisY,
            1.0,
            np.array([np.sqrt(3) / 2, 0.5j]),
            3,
            -0.5,
            0.5,
        ),
        (
            RotationAxisZ,
            2.0,
            np.array([np.sqrt(3) / 2, 0.5]),
            3,
            -0.25,
            0.5,
        ),
    ],
)
def test_hhl_statevector_success_branch(
    rotation_axis: type[RotationAxis],
    coefficient: float,
    input_vector: np.ndarray,
    n_qpe: int,
    simulation_time: float,
    rotation_scalar: float,
) -> None:
    """Test HHL by projecting the statevector onto its successful branch."""
    n_input_qubits = 1
    n_sim_qubits = n_qpe + n_input_qubits + 1

    @guppy
    @no_type_check
    def controlled_hamiltonian_simulation(
        control: qubit,
        state_register: array[qubit, n_input_qubits],
        power: int,
    ) -> None:
        axis = rotation_axis()
        axis.prepare_basis(state_register[0])
        crz(
            control,
            state_register[0],
            angle(coefficient * simulation_time * power),
        )
        axis.restore_basis(state_register[0])

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
    def run_hhl() -> None:
        qs = qarray(n_input_qubits)
        prepare_b(qs)
        clock_reg = qarray(n_qpe)
        ancilla = qarray(1)
        hhl(
            qs,
            clock_reg,
            ancilla[0],
            controlled_hamiltonian_simulation,
            eigenvalue_transform,
        )
        state_output("solution", ancilla[0], qs[0])
        discard_array(ancilla)
        discard_array(clock_reg)
        discard_array(qs)

    pauli_matrices = {
        RotationAxisX: np.array([[0.0, 1.0], [1.0, 0.0]]),
        RotationAxisY: np.array([[0.0, -1.0j], [1.0j, 0.0]]),
        RotationAxisZ: np.array([[1.0, 0.0], [0.0, -1.0]]),
    }
    a_mat = coefficient * pauli_matrices[rotation_axis]
    expected_x = np.linalg.solve(a_mat, input_vector)
    expected_x /= np.linalg.norm(expected_x)

    result = run_hhl.emulator(n_qubits=n_sim_qubits).run()
    states = Quest.extract_states_dict(result.results[0].entries)
    solution_state = states["solution"]
    solution_state.specified_qubits = list(range(1, n_qpe + n_input_qubits + 1))
    projected_solution = project_state_onto_bitstring(
        solution_state,
        [False] * n_qpe + [True],
        new_specified_qubits=[],
    )
    actual_state = projected_solution.state.state

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
