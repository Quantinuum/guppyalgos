"""Guppy function builders used to configure HHL."""

from __future__ import annotations

from typing import no_type_check

import numpy as np
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import array, comptime, frozenarray, nat
from guppylang.std.quantum import cx, qubit, ry

from guppyalgos.algorithms.time_evolution.trotter import (
    cntrl_ham_sim_trotter,
    cntrl_trotter_first_order,
)


@guppy
@no_type_check
def multiplexed_ry[n_controls: nat, n_angles: nat](
    angles: frozenarray[float, n_angles],
    controls: array[qubit, n_controls],
    target: qubit,
) -> None:
    r"""Apply an ancilla-free uniformly controlled ``Ry`` rotation.

    For ``angles[k]`` in half-turns, this applies

    .. math::

        \sum_k |k\rangle\langle k|_{\mathrm{controls}} \otimes
        R_y(\pi\,\mathrm{angles}[k])_{\mathrm{target}}.

    The controls are little-endian and the implementation uses a Gray-code
    decomposition containing only ``Ry`` and ``CX`` gates. ``angles`` must hold one
    entry per control-register basis state, i.e. ``n_angles == 2**n_controls``.

    Args:
        angles: One half-turn rotation angle for every control-register basis state.
        controls: Little-endian control register.
        target: Qubit the multiplexed rotation acts on.

    """
    dim = 2**n_controls
    for gray_index in range(dim):
        gray_code = gray_index ^ (gray_index >> 1)
        rotation = 0.0
        for basis_state in range(dim):
            parity = 0
            for bit in range(n_controls):
                parity += ((gray_code >> bit) & 1) & (
                    (basis_state >> (n_controls - 1 - bit)) & 1
                )
            if parity % 2 == 0:
                rotation += angles[basis_state]
            else:
                rotation -= angles[basis_state]
        ry(target, angle(rotation / float(dim)))

        # Index of the single control bit that differs between consecutive Gray codes.
        toggled_bit = 0
        counter = gray_index + 1
        while counter % 2 == 0 and toggled_bit < n_controls - 1:
            counter //= 2
            toggled_bit += 1
        cx(controls[n_controls - 1 - toggled_bit], target)


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
        multiplexed_ry(comptime(angles), clock_reg, ancilla)

    return eigenvalue_inversion
