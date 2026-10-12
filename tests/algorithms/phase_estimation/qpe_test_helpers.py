"""Shared helpers for phase-estimation tests."""

from __future__ import annotations

from typing import no_type_check

import numpy as np
from guppylang import comptime, guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle, pi
from guppylang.std.builtins import array, nat, output
from guppylang.std.debug import state_output
from guppylang.std.quantum import (
    collect_measurements,
    crz,
    discard_array,
    measure_array,
    qubit,
    ry,
)
from numpy.typing import NDArray

from guppyalgos.algorithms.phase_estimation import iqpe, qpe
from guppyalgos.primitives.subroutines.qft import iqft, qft
from guppyalgos.primitives.state_preparation.uniform import uniform_state
from guppyalgos.utils import dominant_measured_phase as utils_dominant_measured_phase
from guppyalgos.utils import fixed_point_to_float, float_to_fixed_point, qarray

from guppyalgos.testing import switch_endianness


@guppy.struct
class RzUnitaryRegs[n_q_s: nat]:
    """Wrapper for the single-qubit system register used in the Rz QPE tests."""

    system: array[qubit, n_q_s]


def numpy_uniform(n_qubits: int) -> NDArray[np.complex128]:
    """Return the analytical uniformly distributed state."""
    return np.array([1 / (2 ** (n_qubits / 2))] * (2**n_qubits))


def ry_state_probabilities(x: float) -> tuple[float, float]:
    """Return the exact branch probabilities for the two system eigenstates."""
    x = np.pi * x
    return float(np.cos(x / 2) ** 2), float(np.sin(x / 2) ** 2)


def exact_kickback_state(phase: float, n_ancilla: int) -> NDArray[np.complex128]:
    """Return the exact pre-IQFT phase on the kickback register.

    The default little-endian form matches the register order used by the
    current QPE/QFT helpers. This is enforced by the switch_endianness in the return.
    """
    dim = 2**n_ancilla
    basis_index = np.arange(dim)
    state = np.exp(1j * np.pi * np.mod(phase, 2) * basis_index) / np.sqrt(dim)
    return switch_endianness(state)


def exact_kickback_density_matrix(
    phase: float, n_ancilla: int, eigenmix: float
) -> NDArray[np.complex128]:
    """Return the exact reduced density matrix of the kickback register."""
    ground_prob, excited_prob = ry_state_probabilities(eigenmix)
    ground_ramp = exact_kickback_state(phase, n_ancilla)
    excited_ramp = exact_kickback_state(-phase, n_ancilla)
    return ground_prob * np.outer(
        ground_ramp, np.conj(ground_ramp)
    ) + excited_prob * np.outer(excited_ramp, np.conj(excited_ramp))


def phase_key(phase: float, n_ancilla: int) -> str:
    """Return the exact QPE measurement key for a phase with one integer bit."""
    bits = float_to_fixed_point(phase % 2, n_ancilla - 1, int_bits=1)
    return "".join("1" if bit else "0" for bit in bits)


def is_exactly_expressible(phi: float, n_ancilla: int) -> bool:
    """Check that phi is expressible as a single n-bit bitstring."""
    phi_mod_2 = float(np.mod(phi, 2))
    return bool(
        np.isclose(
            phi_mod_2,
            fixed_point_to_float(
                float_to_fixed_point(phi_mod_2, n_ancilla - 1, int_bits=1),
                int_bits=1,
            ),
        )
    )


def make_uniform_ancilla_prep[n_ancilla_q: nat](
    n_ancilla_qubits: int,
) -> GuppyFunctionDefinition[[array[qubit, n_ancilla_q]], None]:
    """Return the ancilla uniform-state factory used by the QPE tests."""
    return uniform_state(2**n_ancilla_qubits)


def make_ry_state[n_state: nat](
    ry_angle: float,
) -> GuppyFunctionDefinition[[array[qubit, n_state]], None]:
    """Return a reusable Ry state-preparation box.

    The same rotation is applied to every qubit in the input register.
    """

    @guppy
    @no_type_check
    def state_preparation[n_state: nat](system_register: array[qubit, n_state]) -> None:
        rotation = angle(comptime(float(ry_angle)))
        for i in range(n_state):
            ry(system_register[i], rotation)

    return state_preparation


def make_rz_power_oracle[n_q_s: nat](
    phi: float,
) -> GuppyFunctionDefinition[[qubit, RzUnitaryRegs[n_q_s], int], None]:
    """Return the controlled-Rz power oracle used by the toy QPE tests."""

    @guppy
    @no_type_check
    def power_oracle[n_q_s: nat](
        ctrl: qubit, unitary_regs: RzUnitaryRegs[n_q_s], power: int
    ) -> None:
        crz(ctrl, unitary_regs.system[0], -2 * pi * phi * power)

    return power_oracle


def make_inverse_rz_power_oracle[n_q_s: nat](
    phi: float,
) -> GuppyFunctionDefinition[[qubit, RzUnitaryRegs[n_q_s], int], None]:
    """Return the inverse controlled-Rz power oracle used in the reversibility test."""

    @guppy
    @no_type_check
    def inverse_power_oracle[n_q_s: nat](
        ctrl: qubit, unitary_regs: RzUnitaryRegs[n_q_s], power: int
    ) -> None:
        crz(ctrl, unitary_regs.system[0], -2 * pi * phi * -power)

    return inverse_power_oracle


def make_simple_qpe_program[n_state: nat](
    n_ancilla: int,
    state_preparation: GuppyFunctionDefinition[[array[qubit, n_state]], None],
    power_oracle: GuppyFunctionDefinition[[qubit, RzUnitaryRegs[n_state], int], None],
) -> GuppyFunctionDefinition[[], None]:
    """Build the canonical measurement program used by the exact Rz QPE tests."""
    ancilla_prep_function = make_uniform_ancilla_prep(n_ancilla)

    @guppy
    @no_type_check
    def abstract_canonical_qpe() -> None:
        unitary_regs = RzUnitaryRegs(qarray(1))
        phase_reg = qarray(n_ancilla)
        state_preparation(unitary_regs.system)
        ancilla_prep_function(phase_reg)
        state_output("phase_uniform_check", phase_reg)
        qpe(phase_reg, unitary_regs, power_oracle)
        state_output("qpe_register", phase_reg)
        qft(phase_reg)
        state_output("kickback_register", phase_reg)
        iqft(phase_reg)
        output("qpe_bitstring", collect_measurements(measure_array(phase_reg)))
        discard_array(unitary_regs.system)

    return abstract_canonical_qpe


def make_simple_qpe_reversibility_program[n_state: nat](
    n_ancilla: int,
    state_preparation: GuppyFunctionDefinition[[array[qubit, n_state]], None],
    power_oracle: GuppyFunctionDefinition[[qubit, RzUnitaryRegs[n_state], int], None],
    inverse_power_oracle: GuppyFunctionDefinition[
        [qubit, RzUnitaryRegs[n_state], int], None
    ],
) -> GuppyFunctionDefinition[[], None]:
    """Build the QPE/inverse-QPE round-trip program used by the Rz tests."""
    ancilla_prep_function = make_uniform_ancilla_prep(n_ancilla)

    @guppy
    @no_type_check
    def abstract_canonical_qpe() -> None:
        unitary_regs = RzUnitaryRegs(qarray(1))
        phase_reg = qarray(n_ancilla)
        state_preparation(unitary_regs.system)
        state_output("unitary_regs_check", unitary_regs.system)
        ancilla_prep_function(phase_reg)
        state_output("phase_uniform_check", phase_reg)
        qpe(phase_reg, unitary_regs, power_oracle)
        iqpe(phase_reg, unitary_regs, inverse_power_oracle)
        state_output("phase_uniform_check_2", phase_reg)
        state_output("unitary_regs_check_2", unitary_regs.system)
        discard_array(phase_reg)
        discard_array(unitary_regs.system)

    return abstract_canonical_qpe


def make_simple_qpe_kickback_program[n_state: nat](
    n_ancilla: int,
    state_preparation: GuppyFunctionDefinition[[array[qubit, n_state]], None],
    power_oracle: GuppyFunctionDefinition[[qubit, RzUnitaryRegs[n_state], int], None],
) -> GuppyFunctionDefinition[[], None]:
    """Build the pre-IQFT kickback program used by the Rz density-matrix test."""
    ancilla_prep_function = make_uniform_ancilla_prep(n_ancilla)

    @guppy
    @no_type_check
    def abstract_kickback_state() -> None:
        unitary_regs = RzUnitaryRegs(qarray(1))
        phase_reg = qarray(n_ancilla)
        state_preparation(unitary_regs.system)
        ancilla_prep_function(phase_reg)
        qpe(phase_reg, unitary_regs, power_oracle)
        qft(phase_reg)
        state_output("kickback_register", phase_reg)
        discard_array(phase_reg)
        discard_array(unitary_regs.system)

    return abstract_kickback_state


def make_trotter_power_oracle[n_trotter_state: nat](
    controlled_step: GuppyFunctionDefinition[
        [qubit, array[qubit, n_trotter_state], float], None
    ],
    time_step: float,
    n_state_qubits: int,
) -> GuppyFunctionDefinition[[qubit, array[qubit, n_trotter_state], int], None]:
    """Return the powered oracle wrapper used by the trotter QPE tests."""

    @guppy
    @no_type_check
    def power_oracle(
        control: qubit,
        state_reg: array[qubit, n_state_qubits],
        power: int,
    ) -> None:
        for _ in range(power):
            controlled_step(control, state_reg, time_step)

    return power_oracle


def make_trotter_qpe_program[n_state: nat](
    n_ancilla: int,
    n_state_qubits: int,
    state_preparation: GuppyFunctionDefinition[[array[qubit, n_state]], None],
    power_oracle: GuppyFunctionDefinition[[qubit, array[qubit, n_state], int], None],
) -> GuppyFunctionDefinition[[], None]:
    """Build the canonical measurement program used by the trotter QPE tests."""
    ancilla_prep_function = make_uniform_ancilla_prep(n_ancilla)

    @guppy
    @no_type_check
    def canonical_trotter_qpe_program() -> None:
        state_reg = qarray(n_state_qubits)
        phase_reg = qarray(n_ancilla)

        state_preparation(state_reg)
        ancilla_prep_function(phase_reg)
        qpe(phase_reg, state_reg, power_oracle)
        output("qpe_bitstring", collect_measurements(measure_array(phase_reg)))
        discard_array(state_reg)

    return canonical_trotter_qpe_program


def make_trotter_qpe_diagnostic_program[n_state: nat](
    n_ancilla: int,
    n_state_qubits: int,
    state_preparation: GuppyFunctionDefinition[[array[qubit, n_state]], None],
    power_oracle: GuppyFunctionDefinition[[qubit, array[qubit, n_state], int], None],
) -> GuppyFunctionDefinition[[], None]:
    """Build a Trotter QPE program with register snapshots and measurement."""
    ancilla_prep_function = make_uniform_ancilla_prep(n_ancilla)

    @guppy
    @no_type_check
    def diagnostic_program() -> None:
        state_reg = qarray(n_state_qubits)
        phase_reg = qarray(n_ancilla)

        state_preparation(state_reg)
        ancilla_prep_function(phase_reg)
        qpe(phase_reg, state_reg, power_oracle)
        state_output("system_register", state_reg)
        state_output("qpe_register", phase_reg)
        qft(phase_reg)
        state_output("kickback_register", phase_reg)
        iqft(phase_reg)
        discard_array(phase_reg)
        discard_array(state_reg)

    return diagnostic_program


def dominant_trotter_measured_phase[n_state: nat](
    n_ancilla: int,
    n_state_qubits: int,
    state_preparation: GuppyFunctionDefinition[[array[qubit, n_state]], None],
    power_oracle: GuppyFunctionDefinition[[qubit, array[qubit, n_state], int], None],
    shots: int = 500,
    seed: int = 5,
) -> float:
    """Run the trotter QPE measurement program and return the dominant phase."""
    measurement_program = make_trotter_qpe_program(
        n_ancilla,
        n_state_qubits,
        state_preparation,
        power_oracle,
    )
    counts = (
        measurement_program.emulator(n_qubits=n_ancilla + n_state_qubits)
        .with_seed(seed)
        .with_shots(shots)
        .run()
        .register_counts()["qpe_bitstring"]
    )
    return utils_dominant_measured_phase(counts)[2]


__all__ = [
    "RzUnitaryRegs",
    "dominant_trotter_measured_phase",
    "exact_kickback_density_matrix",
    "exact_kickback_state",
    "is_exactly_expressible",
    "make_inverse_rz_power_oracle",
    "make_ry_state",
    "make_rz_power_oracle",
    "make_simple_qpe_kickback_program",
    "make_simple_qpe_program",
    "make_simple_qpe_reversibility_program",
    "make_trotter_power_oracle",
    "make_trotter_qpe_diagnostic_program",
    "make_trotter_qpe_program",
    "make_uniform_ancilla_prep",
    "numpy_uniform",
    "phase_key",
    "ry_state_probabilities",
]
