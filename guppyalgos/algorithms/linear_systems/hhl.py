"""The HHL algorithm for solving linear systems of equations [1].

References:
    [1] Harrow, A. W., Hassidim, A., & Lloyd, S. (2009). Quantum algorithm for linear
    systems of equations. Physical review letters, 103(15), 150502.

"""

from __future__ import annotations

from typing import no_type_check

import numpy as np
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import crz, h, measure, qubit, s, sdg, x

from guppyalgos.algorithms.phase_estimation import iqpe, qpe
from guppyalgos.algorithms.time_evolution.trotter import (
    cntrl_ham_sim_trotter,
    cntrl_trotter_first_order,
)
from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx
from guppyalgos.primitives.measurement import discard_array_zero
from guppyalgos.primitives.measurement.utils import discard_zero
from guppyalgos.utils import qarray, transversal


def hhl_power_oracles[n_input: nat](
    input_matrix: zqp.RealTermSum,
    time_step: float,
    n_trotter_steps: int,
    n_input_qubits: int | None = None,
) -> tuple[
    GuppyFunctionDefinition[[qubit, array[qubit, n_input], int], None],
    GuppyFunctionDefinition[[qubit, array[qubit, n_input], int], None],
]:
    """Build the forward and inverse powered controlled-evolution oracles for HHL.

    Args:
        input_matrix: The Hamiltonian matrix represented as a real Pauli term sum.
        time_step: Trotter evolution time step.
        n_trotter_steps: Number of Trotter steps per unit evolution.
        n_input_qubits: Number of qubits in the input register.

    Returns:
        A tuple of (power_oracle, inverse_power_oracle).

    """
    n_qubits = len(input_matrix.qubits) if n_input_qubits is None else n_input_qubits
    cntrl_trotter_step = cntrl_trotter_first_order(
        input_matrix, n_state_qubits=n_qubits
    )

    def _make_power_oracle(
        time_direction: float,
    ) -> GuppyFunctionDefinition[[qubit, array[qubit, n_input], int], None]:
        sim_step = cntrl_ham_sim_trotter(
            cntrl_trotter_step,
            n_trotter_steps,
            time_direction * time_step,
            n_qubits,
        )

        @guppy
        @no_type_check
        def _power_oracle(
            ctrl: qubit,
            unitary_regs: array[qubit, n_qubits],
            power: int,
        ) -> None:
            for _ in range(power):
                sim_step(ctrl, unitary_regs)

        return _power_oracle

    return _make_power_oracle(1.0), _make_power_oracle(-1.0)


def eigenvalue_inversion_angles(
    n_qpe: int,
    time_step: float,
    rotation_scalar: float,
    n_trotter_steps: int = 1,
) -> list[float]:
    """Calculate the half-turn rotation angles for eigenvalue inversion.

    Args:
        n_qpe: Number of clock qubits in QPE.
        time_step: Trotter evolution time step.
        rotation_scalar: Constant C scaling the inversion factor C / lambda.
        n_trotter_steps: Number of Trotter steps per unit time evolution.

    Returns:
        List of 2**n_qpe half-turn angles for Ry rotation on the ancilla qubit.

    """
    dim = 2**n_qpe
    total_time = time_step * n_trotter_steps
    angles = []
    for k in range(dim):
        if k == 0:
            angles.append(0.0)
            continue
        k_signed = k if k < dim // 2 else k - dim
        # One first-order Trotter step is U = exp(-i * pi * total_time / 2 * H).
        # Phase kickback: 2*pi*phi = - pi * total_time * lambda / 2
        # => phi = - total_time * lambda / 4.
        # QPE maps phi to k_signed / dim
        # => lambda = - 4 * k_signed / (dim * total_time).
        # Inversion amplitude ratio = C / lambda
        # => ratio = - (C * dim * total_time) / (4 * k_signed).
        ratio = -(rotation_scalar * dim * total_time) / (4.0 * k_signed)
        ratio = max(-1.0, min(1.0, ratio))
        half_turn = (2.0 / np.pi) * np.arcsin(ratio)
        angles.append(float(half_turn))
    return angles


def hhl_conditional_rotation[n_clock: nat](
    n_qpe: int,
    angles: list[float],
) -> GuppyFunctionDefinition[[array[qubit, n_clock], qubit], None]:
    r"""Build the clock-conditioned Ry rotation acting on the ancilla qubit.

    The unitary applies a state-dependent rotation around the Y axis:

    .. math::

        U = \sum_{k=0}^{2^{n_{\text{qpe}}} - 1} |k\rangle\langle k|_{\text{clock}}
        \otimes R_y(a_k \pi)_{\text{ancilla}}

    where $a_k = \text{angles}[k]$ is the rotation angle in half-turns.
    Acting on an ancilla initialized in $|0\rangle$, this implements the
    transformation:

    .. math::

        |k\rangle_{\text{clock}} |0\rangle_{\text{ancilla}} \mapsto
        |k\rangle_{\text{clock}} \left(
            \cos\left(\frac{a_k \pi}{2}\right) |0\rangle_{\text{ancilla}}
            + \sin\left(\frac{a_k \pi}{2}\right) |1\rangle_{\text{ancilla}}
        \right).

    Args:
        n_qpe: Number of clock qubits.
        angles: List of half-turn rotation angles indexed by clock basis state.

    Returns:
        A Guppy function applying the conditioned rotation.

    """
    active_rotations = [
        (k, angle_val) for k, angle_val in enumerate(angles) if abs(angle_val) > 1e-12
    ]

    if n_qpe == 1:

        @guppy
        @no_type_check
        def rot_1(clock_reg: array[qubit, 1], ancilla: qubit) -> None:
            for k, a_k in active_rotations:
                if (k & 1) == 0:
                    x(clock_reg[0])
                sdg(ancilla)
                h(ancilla)
                crz(clock_reg[0], ancilla, angle(a_k))
                h(ancilla)
                s(ancilla)
                if (k & 1) == 0:
                    x(clock_reg[0])

        return rot_1

    @guppy
    @no_type_check
    def rot_n(clock_reg: array[qubit, n_qpe], ancilla: qubit) -> None:
        flag = qubit()
        for k, a_k in active_rotations:
            for b in range(n_qpe):
                if not ((k >> b) & 1):
                    x(clock_reg[b])

            cnx(clock_reg, flag)
            sdg(ancilla)
            h(ancilla)
            crz(flag, ancilla, angle(a_k))
            h(ancilla)
            s(ancilla)
            cnx(clock_reg, flag)

            for b in range(n_qpe):
                if not ((k >> b) & 1):
                    x(clock_reg[b])

        discard_zero(flag)

    return rot_n


def hhl[n_input: nat](
    input_matrix: zqp.RealTermSum,
    n_qpe: int,
    rotation_scalar: float,
    time_step: float = 1.0,
    n_trotter_steps: int = 1,
    n_input_qubits: int | None = None,
) -> GuppyFunctionDefinition[[array[qubit, n_input]], bool]:
    """Construct a Guppy function for the HHL algorithm.

    Args:
        input_matrix: The Hermitian matrix A represented as a real Pauli term sum.
        n_qpe: Number of qubits in the clock register for quantum phase estimation.
        rotation_scalar: Constant scaling the inverse eigenvalue rotation.
        time_step: Time step for Trotterized Hamiltonian simulation.
        n_trotter_steps: Number of Trotter steps per unit evolution.
        n_input_qubits: Number of qubits in the input/system register.

    Returns:
        A Guppy function that applies HHL to an input state register and returns
        a boolean flag indicating whether the algorithm succeeded (ancilla measured 1).

    """
    n_qubits = len(input_matrix.qubits) if n_input_qubits is None else n_input_qubits
    power_oracle, inverse_power_oracle = hhl_power_oracles(
        input_matrix=input_matrix,
        time_step=time_step,
        n_trotter_steps=n_trotter_steps,
        n_input_qubits=n_qubits,
    )

    angles = eigenvalue_inversion_angles(
        n_qpe=n_qpe,
        time_step=time_step,
        rotation_scalar=rotation_scalar,
        n_trotter_steps=n_trotter_steps,
    )

    clock_rotation = hhl_conditional_rotation(
        n_qpe=n_qpe,
        angles=angles,
    )

    @guppy
    @no_type_check
    def hhl_fn(input_state: array[qubit, n_qubits]) -> bool:
        clock_reg = qarray(n_qpe)

        transversal(h, clock_reg)
        qpe(clock_reg, input_state, power_oracle)

        ancilla = qubit()
        clock_rotation(clock_reg, ancilla)

        iqpe(clock_reg, input_state, inverse_power_oracle)
        transversal(h, clock_reg)

        discard_array_zero(clock_reg)
        success = measure(ancilla).read()

        return success

    return hhl_fn
