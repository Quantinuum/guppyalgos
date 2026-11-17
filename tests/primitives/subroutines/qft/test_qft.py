"""QFT tests."""

import pytest
import numpy as np
from typing import no_type_check

from guppylang.decorator import guppy
from guppylang.std.builtins import array, control, dagger
from guppylang.std.debug import state_output
from guppylang.std.quantum import qubit, x
from numpy.typing import NDArray
from selene_sim import QuantumReplay, Quest

from guppyalgos.primitives.state_preparation.uniform import uniform_state
from guppyalgos.primitives.subroutines.qft import qft
from guppyalgos.testing import (
    Endianness,
    assert_allclose_ignorephase,
    assert_unitary_modifiers,
    get_statevector,
    get_unitary,
    get_unitary_assumed_phase,
    switch_endianness,
    _statevector_with_zeroed_ancilla,
)
from guppyalgos.utils import qarray
from guppyalgos.utils.guppy.array import join_arrays, split_array
from guppyalgos.utils.guppy.unsafe_borrow import (
    _unsafe_array_borrow,
    _unsafe_array_unborrow,
)
from guppylang.std.quantum import discard_array, h


def qft_unitary_algebra(n_qubits: int) -> NDArray[np.complex128]:
    """Return the unitary matrix for the n qubit Quantum Fourier transform."""
    N = 2**n_qubits
    omega = np.exp(2j * np.pi / N)
    QFT = np.zeros((N, N), dtype=complex)

    for j in range(N):
        for k in range(N):
            QFT[j, k] = omega ** (j * k)

    return QFT / np.sqrt(N)


@pytest.mark.parametrize("n_qubits", [1, 3, 4])
@no_type_check
def test_qft_unitary(n_qubits: int) -> None:
    """Test QFT and its public modifiers against the QFT unitary matrix."""
    qft_arr = qft_unitary_algebra(n_qubits)

    # Verify unitarity of the independent reference matrix.
    identity = np.eye(2**n_qubits)
    product = np.dot(qft_arr, qft_arr.conj().T)
    assert_allclose_ignorephase(product, identity)

    if n_qubits == 1:

        @guppy
        def main(state_qreg: array[qubit, n_qubits]) -> None:
            h(state_qreg[0])

        guppy_u = get_unitary_assumed_phase(
            main, n_qubits, endianness=Endianness.LITTLE
        )
        assert_allclose_ignorephase(guppy_u, qft_arr.conj())

    @guppy(unitary=True)
    def circuit(qs: array[qubit, n_qubits]) -> None:
        qft(qs)

    assert_unitary_modifiers(
        circuit,
        n_qubits,
        expected_unitary=qft_arr,
        endianness=Endianness.LITTLE,
        n_extra_qubits=0 if n_qubits == 1 else 1,
    )


@pytest.mark.parametrize("n_qubits", [1, 3, 4])
@no_type_check
def test_daggered_qft_sv(n_qubits: int) -> None:
    """Test ``with dagger`` transforms a uniform superposition to |0>."""
    uniform_state_prep = uniform_state(2**n_qubits)

    @guppy
    def main() -> None:
        qs = qarray(n_qubits)
        uniform_state_prep(qs)
        with dagger:
            qft(qs)
        state_output("result_state", qs)
        discard_array(qs)

    simulated_state = get_statevector(main, n_qubits)
    expected_state = np.zeros(simulated_state.shape, dtype=complex)
    expected_state[0] = 1.0
    assert_allclose_ignorephase(simulated_state, expected_state)


@pytest.mark.parametrize("n_qubits", [1, 3, 4])
@no_type_check
def test_qft_sv(n_qubits: int) -> None:
    """Test the QFT U by transforming |0> to a uniform superposition."""

    @guppy
    def main2() -> None:
        qs = qarray(n_qubits)
        qft(qs)
        state_output("result_state", qs)
        discard_array(qs)

    expected_state = np.array([1 / (2 ** (n_qubits / 2))] * (2**n_qubits))
    simulated_state = get_statevector(main2, n_qubits)
    assert_allclose_ignorephase(simulated_state, expected_state)


@pytest.mark.parametrize("n_qubits", [3, 4])
@pytest.mark.parametrize("k", [1, 3, 5])
@no_type_check
def test_qft_basis_state(n_qubits: int, k: int) -> None:
    """Test that QFT correctly transforms a computational basis state |k⟩."""
    N = 2**n_qubits
    omega = np.exp(2j * np.pi / N)

    @guppy
    def main() -> None:
        qs = qarray(n_qubits)
        for i in range(n_qubits):
            # little-endian bit check
            if (k >> i) & 1:
                x(qs[i])

        qft(qs)
        state_output("result_state", qs)
        discard_array(qs)

    simulated = get_statevector(main, n_qubits)

    expected = np.array(
        [omega ** (j * k) for j in range(N)],
        dtype=complex,
    ) / np.sqrt(N)
    expected = switch_endianness(expected)

    assert_allclose_ignorephase(simulated, expected)


def _multi_controlled_qft_unitary(
    n_controls: int, n_targets: int
) -> NDArray[np.complex128]:
    """Return QFT controlled by the all-one control-register state."""
    control_size = 2**n_controls
    target_size = 2**n_targets
    control_on = np.zeros((control_size, control_size), dtype=np.complex128)
    control_on[-1, -1] = 1
    identity = np.eye(target_size, dtype=np.complex128)
    return np.kron(identity, np.eye(control_size) - control_on) + np.kron(
        qft_unitary_algebra(n_targets), control_on
    )


@pytest.mark.parametrize("n_controls", [2, 3])
@pytest.mark.parametrize("daggered", [False, True])
@no_type_check
def test_qft_multiple_controls(n_controls: int, daggered: bool) -> None:
    """Check multi-control works for QFT and its dagger."""
    n_targets = 2
    n_qubits = n_controls + n_targets

    @guppy
    @no_type_check
    def circuit(qs: array[qubit, n_qubits]) -> None:
        controls, targets = split_array(_unsafe_array_borrow(qs), n_controls, n_targets)
        with control(controls):
            if daggered:
                with dagger:
                    qft(targets)
            else:
                qft(targets)
        _unsafe_array_unborrow(qs, join_arrays(controls, targets, n_qubits))

    actual = get_unitary(
        circuit,
        n_qubits,
        endianness=Endianness.LITTLE,
        # The custom control path uses one conjunction flag and its CNX workspace;
        # each QFT phase additionally combines that flag with its intrinsic control.
        n_extra_qubits=n_controls,
    )
    assert_allclose_ignorephase(
        actual,
        (
            _multi_controlled_qft_unitary(n_controls, n_targets).conj().T
            if daggered
            else _multi_controlled_qft_unitary(n_controls, n_targets)
        ),
    )


@pytest.mark.parametrize("daggered", [False, True])
@no_type_check
def test_qft_three_controls_preserve_coherence(daggered: bool) -> None:
    """Check three-controlled QFT preserves off/on control coherence."""
    n_controls, n_targets = 3, 2
    n_qubits = n_controls + n_targets

    @guppy
    @no_type_check
    def circuit() -> None:
        controls = qarray(n_controls)
        targets = qarray(n_targets)

        # Prepare (|000> + |111>) / sqrt(2).
        h(controls[0])
        with control(controls[0]):
            x(controls[1])
            x(controls[2])

        # |01> distinguishes forward QFT from its dagger.
        x(targets[0])

        with control(controls):
            if daggered:
                with dagger:
                    qft(targets)
            else:
                qft(targets)

        # Join only to output the complete logical state.
        result = join_arrays(controls, targets, n_qubits)
        state_output("result_state", result)
        discard_array(result)

    initial = np.zeros(2**n_qubits, dtype=np.complex128)
    initial[[0b01000, 0b01111]] = 1 / np.sqrt(2)

    controlled_qft = _multi_controlled_qft_unitary(n_controls, n_targets)
    expected = (
        controlled_qft.conj().T @ initial if daggered else controlled_qft @ initial
    )

    replay = QuantumReplay(
        simulator=Quest(random_seed=17),
        resume_with_measurement=True,
        measurements=[[False], [True]],
    )
    result = (
        circuit.emulator(n_qubits + n_controls)
        .with_simulator(replay)
        .with_shots(2)
        .run()
    )

    for shot in result.results:
        states = Quest.extract_states_dict(shot.entries)
        actual = _statevector_with_zeroed_ancilla(states["result_state"])
        assert_allclose_ignorephase(actual, expected)
