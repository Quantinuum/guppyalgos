"""Tests for the HHL algorithm."""

from __future__ import annotations
from guppylang.defs import GuppyFunctionDefinition

from typing import no_type_check

import numpy as np
import pytest
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.std.builtins import array, nat
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard, discard_array, qubit
from selene_sim import Quest

from guppyalgos.algorithms.linear_systems import (
    create_controlled_hamiltonian_simulation,
    create_eigenvalue_inversion,
    hhl,
)
from guppyalgos.algorithms.linear_systems.hhl_utils import eigenvalue_inversion_angles
from guppyalgos.algorithms.state_preparation import multiplexor_prep
from guppyalgos.utils import apply_bitstring, int_to_bits, qarray, register_size
from guppyalgos.primitives.measurement import discard_array_zero
from tests.helpers import assert_allclose_ignorephase, switch_endianness


def test_eigenvalue_inversion_angles_use_signed_clock_labels() -> None:
    """The inversion angles encode C divided by each signed clock label."""
    angles = eigenvalue_inversion_angles(n_qpe=3, rotation_scalar=1.0)

    expected = [
        0.0,
        1.0,
        1.0 / 3.0,
        2.0 / np.pi * np.arcsin(1.0 / 3.0),
        -2.0 / np.pi * np.arcsin(1.0 / 4.0),
        -2.0 / np.pi * np.arcsin(1.0 / 3.0),
        -1.0 / 3.0,
        -1.0,
    ]

    assert angles == pytest.approx(expected)


@pytest.mark.parametrize(
    (
        "ham_str",
        "n_input_qubits",
        "input_vector",
        "n_qpe",
        "time_step",
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
    time_step: float,
    rotation_scalar: float,
) -> None:
    """Test HHL across 1-, 2-, and 3-qubit linear systems with repeat-until-success."""
    ham_op = zqp.RealTermSum.from_str(ham_str)

    n_sim_qubits = n_qpe + n_input_qubits + 1
    controlled_hamiltonian_simulation = create_controlled_hamiltonian_simulation(
        ham_op, time_step, n_input_qubits=n_input_qubits
    )
    eigenvalue_inversion = create_eigenvalue_inversion(n_qpe, rotation_scalar)
    hhl_op = hhl(controlled_hamiltonian_simulation, eigenvalue_inversion)
    prepare_b = multiplexor_prep(input_vector)

    @guppy
    @no_type_check
    def run_hhl_rus() -> None:
        while True:
            qs = qarray(n_input_qubits)
            prepare_b(qs)
            success = hhl_op(qs)
            if success:
                state_output("solution", qs)
                discard_array(qs)
                break
            discard_array(qs)

    sim_result = run_hhl_rus.emulator(n_qubits=n_sim_qubits).run()

    states = Quest.extract_states_dict(sim_result.results[0].entries)
    actual_state = switch_endianness(
        states["solution"].get_state_vector_distribution()[0].state
    )

    a_mat = ham_op.to_sparse_matrix(False).toarray()
    expected_x = np.linalg.solve(a_mat, input_vector)
    expected_x /= np.linalg.norm(expected_x)

    assert_allclose_ignorephase(actual_state, expected_x, threshold=1e-5)


@pytest.mark.parametrize(
    ("eigenvalue_inversion", "scaling_factor", "clock_reg_state", "n_ancillas"),
    [
        (create_eigenvalue_inversion(2, 1.0), 1.0, 0.5, 0),
        (create_eigenvalue_inversion(3, 0.1), 0.1, 0.6, 0),
    ],
)
def test_eigenvalue_inversion[n_clock: nat](
    eigenvalue_inversion: GuppyFunctionDefinition[[array[qubit, n_clock], qubit], None],
    scaling_factor: float,
    clock_reg_state: float,
    n_ancillas: int,
) -> None:
    """Test that the eigenvalue inversion function has the expected behaviour.

    For an n-qubit clock basis state |k⟩, interpret k as the signed integer label
    λ = k for k < 2^(n - 1), and λ = k - 2^n otherwise. The inversion should perform
    |k⟩|0⟩ → |k⟩(√(1 - |C/λ|^2)|0⟩ + (C/λ)|1⟩), where C is expressed in the same
    integer-label units as λ.

    """
    clock_reg_size = register_size(eigenvalue_inversion, 0)
    clock_reg_state_int = int(clock_reg_state * (2**clock_reg_size))
    clock_reg_state_bits = int_to_bits(clock_reg_state_int, clock_reg_size)
    signed_clock_label = (
        clock_reg_state_int
        if clock_reg_state_int < 2 ** (clock_reg_size - 1)
        else clock_reg_state_int - 2**clock_reg_size
    )
    expected_amplitude = scaling_factor / signed_clock_label
    expected_state = np.array(
        [np.sqrt(1.0 - expected_amplitude**2), expected_amplitude]
    )

    @guppy
    @no_type_check
    def main() -> None:
        clock_reg = qarray(clock_reg_size)
        apply_bitstring(clock_reg, clock_reg_state_bits)

        ancilla = qubit()
        eigenvalue_inversion(clock_reg, ancilla)

        apply_bitstring(clock_reg, clock_reg_state_bits)
        discard_array_zero(clock_reg)

        state_output("ancilla", ancilla)
        discard(ancilla)

    sim_result = main.emulator(n_qubits=clock_reg_size + n_ancillas + 1).run()
    states = Quest.extract_states_dict(sim_result.results[0].entries)
    actual_state = states["ancilla"].get_state_vector_distribution()[0].state

    assert_allclose_ignorephase(actual_state, expected_state)
