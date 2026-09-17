"""Guppy function builders used to configure HHL."""

from __future__ import annotations

from typing import no_type_check

import numpy as np
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.quantum import qubit

from guppyalgos.algorithms.time_evolution.trotter import (
    cntrl_ham_sim_trotter,
    cntrl_trotter_first_order,
)
from guppyalgos.primitives.rotations import RotationAxisY, multiplexed_rotation


def eigenvalue_inversion_angles(
    n_qpe: int,
    rotation_scalar: float,
) -> list[float]:
    """Calculate the half-turn rotation angles for eigenvalue inversion.

    Args:
        n_qpe: Number of clock qubits in QPE.
        rotation_scalar: Constant C scaling the inversion factor C / lambda.

    Returns:
        List of 2**n_qpe half-turn angles for Ry rotation on the ancilla qubit.

    """
    dim = 2**n_qpe
    angles = []
    for k in range(dim):
        if k == 0:
            angles.append(0.0)
            continue
        k_signed = k if k < dim // 2 else k - dim
        ratio = rotation_scalar / k_signed
        ratio = max(-1.0, min(1.0, ratio))
        half_turn = (2.0 / np.pi) * np.arcsin(ratio)
        angles.append(float(half_turn))
    return angles


def create_controlled_hamiltonian_simulation[n_input: nat](
    input_matrix: zqp.RealTermSum,
    time_step: float,
    n_trotter_steps: int = 1,
    n_input_qubits: int | None = None,
) -> GuppyFunctionDefinition[[qubit, array[qubit, n_input], int], None]:
    """Build a signed-powered controlled Hamiltonian simulation for HHL.

    Args:
        input_matrix: Hamiltonian represented as a real Pauli term sum.
        time_step: Trotter evolution time step.
        n_trotter_steps: Number of Trotter steps per unit evolution.
        n_input_qubits: Number of qubits in the system register. Defaults to the
            number of qubits represented by ``input_matrix``.

    Returns:
        A controlled simulation accepting a signed integer power. Positive powers
        apply forward evolution and negative powers apply inverse evolution.

    """
    n_qubits = len(input_matrix.qubits) if n_input_qubits is None else n_input_qubits
    controlled_trotter_step = cntrl_trotter_first_order(
        input_matrix, n_state_qubits=n_qubits
    )
    forward_simulation = cntrl_ham_sim_trotter(
        controlled_trotter_step, n_trotter_steps, time_step, n_qubits
    )
    inverse_simulation = cntrl_ham_sim_trotter(
        controlled_trotter_step, n_trotter_steps, -time_step, n_qubits
    )

    @guppy
    @no_type_check
    def controlled_hamiltonian_simulation(
        control: qubit,
        state_register: array[qubit, n_qubits],
        power: int,
    ) -> None:
        if power >= 0:
            for _ in range(power):
                forward_simulation(control, state_register)
        else:
            for _ in range(-power):
                inverse_simulation(control, state_register)

    return controlled_hamiltonian_simulation


def create_eigenvalue_inversion[n_clock: nat](
    n_qpe: int,
    rotation_scalar: float,
) -> GuppyFunctionDefinition[[array[qubit, n_clock], qubit], None]:
    """Build the clock-conditioned eigenvalue-inversion rotation for HHL.

    Args:
        n_qpe: Number of clock qubits in QPE.
        rotation_scalar: Constant C scaling the inversion factor C / lambda.

    Returns:
        A Guppy function applying the conditioned rotation to a zeroed ancilla.

    """
    angles = eigenvalue_inversion_angles(n_qpe, rotation_scalar)

    @guppy
    @no_type_check
    def eigenvalue_inversion(clock_reg: array[qubit, n_qpe], ancilla: qubit) -> None:
        multiplexed_rotation(RotationAxisY(), comptime(angles), clock_reg, ancilla)

    return eigenvalue_inversion
