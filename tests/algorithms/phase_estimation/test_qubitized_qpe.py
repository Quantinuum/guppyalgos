"""Canonical phase-estimation tests using a qubitization walk."""

from typing import no_type_check

import numpy as np
import pytest
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.emulator import EmulatorResult
from guppylang.std.angles import angle
from guppylang.std.builtins import array, output
from guppylang.std.debug import state_output
from guppylang.std.quantum import (
    collect_measurements,
    discard_array,
    h,
    measure_array,
    qubit,
    ry,
    toffoli,
    x,
)
from pytest_lazy_fixtures import lf as lazy_fixture
from selene_sim import Quest

from guppyalgos.primitives.gate_decompositions.cnx.cnx import cnx
from guppyalgos.algorithms.block_encoding.lcu import (
    LCUCntrl,
    LCUData,
    build_cntrl_unary_iteration_select,
)
from guppyalgos.algorithms.phase_estimation import (
    QubitizationRegs,
    qubitized_power_oracle,
    qpe,
)
from guppyalgos.algorithms.block_encoding.qubitization import QubitizationCntrl
from guppyalgos.primitives.subroutines.reflection import ReflectionCntrl
from guppyalgos.algorithms.state_preparation import multiplexor_prep
from guppyalgos.utils import (
    binary_fraction,
    phase_distance_mod_2,
    qarray,
    phase_to_energy_qubitized_qpe,
    transversal,
)

dagger = object()


def _phase_distribution(
    result: EmulatorResult, n_phase_qubits: int
) -> dict[float, float]:
    """Extract nonzero phase probabilities from a Selene state snapshot."""
    entries = result.results[0].entries
    phase_state = Quest.extract_states_dict(entries)["phase"]
    probabilities = np.real(np.diag(phase_state.get_density_matrix()))
    return {
        binary_fraction([bit == "1" for bit in f"{index:0{n_phase_qubits}b}"]): float(
            probability
        )
        for index, probability in enumerate(probabilities)
        if probability > 1e-10
    }


def build_qubitized_power_oracle() -> GuppyFunctionDefinition[
    [
        qubit,
        QubitizationRegs[
            1,
            array[qubit, 1],
        ],
        int,
    ],
    None,
]:
    """Build the controlled walk and its QPE power oracle."""

    @guppy
    @no_type_check
    def prepare(prep: array[qubit, 1]) -> None:
        h(prep[0])

    @guppy
    @no_type_check
    def cntrl_select(
        control: qubit,
        prep: array[qubit, 1],
        target: array[qubit, 1],
    ) -> None:
        x(prep[0])
        toffoli(control, prep[0], target[0])
        x(prep[0])
        h(target[0])
        toffoli(control, prep[0], target[0])
        h(target[0])

    @guppy
    @no_type_check
    def cntrl_walk(
        control: qubit,
        prep: array[qubit, 1],
        target: array[qubit, 1],
    ) -> None:
        cntrl_lcu = LCUCntrl(prepare, cntrl_select, prepare)
        reflection = ReflectionCntrl[1](cnx)
        QubitizationCntrl(cntrl_lcu, reflection).compose(control, prep, target)

    @guppy
    @no_type_check
    def power_oracle(
        control: qubit,
        regs: QubitizationRegs[1, array[qubit, 1]],
        power: int,
    ) -> None:
        qubitized_power_oracle(control, regs, power, cntrl_walk)

    return power_oracle


def test_qubitized_qpe_exact_phases() -> None:
    r"""Resolve the eigenvalue of :math:`H=(X+Z)/2` with one PREPARE qubit."""
    power_oracle = build_qubitized_power_oracle()

    @guppy
    @no_type_check
    def main() -> None:
        phase = qarray(3)
        prep = qarray(1)
        target = qarray(1)

        ry(target[0], angle(0.25))
        transversal(h, phase)
        regs = QubitizationRegs(prep, target)
        qpe(phase, regs, power_oracle)

        state_output("phase", phase)
        discard_array(phase)
        discard_array(regs.prep)
        discard_array(regs.target)

    result = main.emulator(n_qubits=5).run()
    phase_distribution = _phase_distribution(result, 3)
    assert phase_distribution == pytest.approx({0.75: 0.5, 1.25: 0.5})

    expected_eigenvalue = 1 / np.sqrt(2)
    for phase in phase_distribution:
        assert phase_to_energy_qubitized_qpe(phase, 1.0) == pytest.approx(
            expected_eigenvalue
        )


def test_qubitized_qpe_off_grid_phases() -> None:
    """Resolve an off-grid eigenphase of :math:`H=(X+Z)/2`."""
    n_phase_qubits = 2
    n_shots = 50
    ideal_phases = {0.75, 1.25}
    grid_spacing = 2 / (2**n_phase_qubits)
    phase_resolution = grid_spacing / 2
    phase_grid = {index * grid_spacing for index in range(2**n_phase_qubits)}
    power_oracle = build_qubitized_power_oracle()

    @guppy
    @no_type_check
    def main() -> None:
        phase = qarray(n_phase_qubits)
        prep = qarray(1)
        target = qarray(1)

        ry(target[0], angle(0.25))
        transversal(h, phase)
        regs = QubitizationRegs(prep, target)
        qpe(phase, regs, power_oracle)

        output("qpe_bitstring", collect_measurements(measure_array(phase)))
        discard_array(regs.prep)
        discard_array(regs.target)

    phase_counts = (
        main.emulator(n_qubits=4)
        .with_seed(5)
        .with_shots(n_shots)
        .run()
        .register_counts()["qpe_bitstring"]
    )
    sampled_phase_counts = {
        binary_fraction([bit == "1" for bit in bitstring]): count
        for bitstring, count in phase_counts.items()
    }
    dominant_measured_phase = max(
        sampled_phase_counts, key=sampled_phase_counts.__getitem__
    )

    assert ideal_phases.isdisjoint(phase_grid)
    assert sum(sampled_phase_counts.values()) == n_shots
    assert (
        min(
            phase_distance_mod_2(dominant_measured_phase, ideal_phase)
            for ideal_phase in ideal_phases
        )
        <= phase_resolution
    )


@pytest.mark.parametrize(
    "ham_op",
    [
        lazy_fixture("ham_3q_posreal_0"),
        lazy_fixture("ham_3q_posreal_1"),
        lazy_fixture("ham_3q_negreal_0"),
    ],
)
def test_qubitized_qpe_unary_iteration_select(ham_op: zqp.RealTermSum) -> None:
    """Resolve eigenphases for Hamiltonians requiring unary-iteration SELECT."""
    n_phase_qubits = 4
    n_shots = 50
    data = LCUData.from_hamiltonian(ham_op)
    # LCU state preparation and SELECT operators
    lcu_prepare = multiplexor_prep(data.amplitudes)
    cntrl_select = build_cntrl_unary_iteration_select(data)
    n_prep_qubits = data.n_prep_qubits
    n_state_qubits = data.n_state_qubits
    hamiltonian = ham_op.to_sparse_matrix(False).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    # Choose the first eigenvalue/eigenvector pair, which is the ground state of
    # this Hamiltonian.
    expected_eigenvalue = float(eigenvalues[0])
    initial_state = eigenvectors[:, 0]
    # Target state preparation
    state_prepare = multiplexor_prep(initial_state)

    assert data.n_terms == 4
    assert n_prep_qubits == 2
    # Verify H|psi> = lambda |psi>.
    assert np.allclose(hamiltonian @ initial_state, expected_eigenvalue * initial_state)

    @guppy
    @no_type_check
    def unprepare(prep: array[qubit, n_prep_qubits]) -> None:
        with dagger:
            lcu_prepare(prep)

    @guppy
    @no_type_check
    def cntrl_walk(
        control: qubit,
        prep: array[qubit, n_prep_qubits],
        target: array[qubit, n_state_qubits],
    ) -> None:
        cntrl_lcu = LCUCntrl(lcu_prepare, cntrl_select, unprepare)
        reflection = ReflectionCntrl[n_prep_qubits](cnx)
        QubitizationCntrl(cntrl_lcu, reflection).compose(control, prep, target)

    @guppy
    @no_type_check
    def power_oracle(
        control: qubit,
        regs: QubitizationRegs[
            n_prep_qubits,
            array[qubit, n_state_qubits],
        ],
        power: int,
    ) -> None:
        qubitized_power_oracle(control, regs, power, cntrl_walk)

    @guppy
    @no_type_check
    def main() -> None:
        phase = qarray(n_phase_qubits)
        prep = qarray(n_prep_qubits)
        target = qarray(n_state_qubits)

        state_prepare(target)
        transversal(h, phase)
        regs = QubitizationRegs(prep, target)
        qpe(phase, regs, power_oracle)

        output("qpe_bitstring", collect_measurements(measure_array(phase)))
        discard_array(regs.prep)
        discard_array(regs.target)

    ideal_phase = float(np.arccos(-expected_eigenvalue / data.l1_norm) / np.pi)
    ideal_phases = {ideal_phase, 2 - ideal_phase}
    phase_resolution = 1 / (2**n_phase_qubits)
    phase_counts = (
        main.emulator(
            n_qubits=n_phase_qubits + n_prep_qubits + n_state_qubits + n_prep_qubits
        )
        .with_seed(5)
        .with_shots(n_shots)
        .run()
        .register_counts()["qpe_bitstring"]
    )
    sampled_phase_counts = {
        binary_fraction([bit == "1" for bit in bitstring]): count
        for bitstring, count in phase_counts.items()
    }
    dominant_measured_phase = max(
        sampled_phase_counts, key=sampled_phase_counts.__getitem__
    )

    assert sum(sampled_phase_counts.values()) == n_shots
    assert (
        min(
            phase_distance_mod_2(dominant_measured_phase, phase)
            for phase in ideal_phases
        )
        <= phase_resolution
    )
