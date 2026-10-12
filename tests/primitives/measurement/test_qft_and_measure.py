"""Dynamic QFT tests."""

import pytest
import numpy as np
from typing import no_type_check
from guppylang.decorator import guppy
from guppylang.std.builtins import array, result

from guppyalgos.primitives.measurement import iqft_and_measure, qft_and_measure
from guppyalgos.primitives.subroutines.qft import iqft, qft
from guppyalgos.utils import qarray, apply_bitstring, int_to_bits, bits_to_int
from guppylang.std.quantum import discard_array, h, cx, s, qubit
from guppylang.std.debug import state_output
from guppyalgos.testing import get_statevector, switch_endianness


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5])
def test_qft_and_measure_basis_state(n_qubits: int) -> None:
    """QFT+measurement maps QFT†|k⟩ onto k."""

    @guppy
    @no_type_check
    def main(k_bits: array[bool, n_qubits]) -> None:
        qs = qarray(n_qubits)
        apply_bitstring(qs, k_bits)

        iqft(qs)
        bits = qft_and_measure(qs)

        for i in range(n_qubits):
            result("qft", bits[i])

        discard_array(qs)

    inputs = [
        {"k_bits": [bool(bit) for bit in int_to_bits(k, n_qubits)]}
        for k in range(2**n_qubits)
    ]

    shots = (
        main.emulator(n_qubits=n_qubits)
        .statevector_sim()
        .with_seed(42)
        .run_per_shot(inputs)
    )

    observed = [
        bits_to_int([int(value) for tag, value in shot.entries if tag == "qft"])
        for shot in shots.results
    ]

    assert observed == list(range(2**n_qubits))


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5])
def test_iqft_and_measure_basis_state(n_qubits: int) -> None:
    """iQFT+measurement maps QFT|k⟩ onto k."""

    @guppy
    @no_type_check
    def main(k_bits: array[bool, n_qubits]) -> None:
        qs = qarray(n_qubits)
        apply_bitstring(qs, k_bits)

        qft(qs)
        bits = iqft_and_measure(qs)

        for i in range(n_qubits):
            result("iqft", bits[i])

        discard_array(qs)

    inputs = [
        {"k_bits": [bool(bit) for bit in int_to_bits(k, n_qubits)]}
        for k in range(2**n_qubits)
    ]

    shots = (
        main.emulator(n_qubits=n_qubits)
        .statevector_sim()
        .with_seed(42)
        .run_per_shot(inputs)
    )

    observed = [
        bits_to_int([int(value) for tag, value in shot.entries if tag == "iqft"])
        for shot in shots.results
    ]

    assert observed == list(range(2**n_qubits))


@guppy
def prepare_nonbasis_state(qs: array[qubit, 3]) -> None:
    """Prepare an entangled test state."""
    h(qs[0])
    s(qs[0])

    h(qs[1])
    cx(qs[1], qs[2])


@pytest.mark.parametrize("inverse", [False, True])
@no_type_check
def test_qft_measurement_channel(inverse: bool) -> None:
    """Dynamic and unitary Fourier measurement channels agree."""
    n_qubits = 3
    n_shots = 10_000

    if inverse:

        @guppy
        def reference() -> None:
            qs = qarray(n_qubits)
            prepare_nonbasis_state(qs)
            iqft(qs)
            state_output("result_state", qs)
            discard_array(qs)

        @guppy
        def dynamic() -> None:
            qs = qarray(n_qubits)
            prepare_nonbasis_state(qs)
            bits = iqft_and_measure(qs)

            for i in range(n_qubits):
                result("fourier", bits[i])

            discard_array(qs)

    else:

        @guppy
        def reference() -> None:
            qs = qarray(n_qubits)
            prepare_nonbasis_state(qs)
            qft(qs)
            state_output("result_state", qs)
            discard_array(qs)

        @guppy
        def dynamic() -> None:
            qs = qarray(n_qubits)
            prepare_nonbasis_state(qs)
            bits = qft_and_measure(qs)

            for i in range(n_qubits):
                result("fourier", bits[i])

            discard_array(qs)

    unitary_state = get_statevector(reference, n_qubits)

    # get_statevector uses simulator ordering; convert back to the logical
    # little-endian ordering
    expected_probabilities = switch_endianness(np.abs(unitary_state) ** 2)

    shots = (
        dynamic.emulator(n_qubits=n_qubits)
        .statevector_sim()
        .with_seed(1234)
        .with_shots(n_shots)
        .run()
    )

    outcomes = [
        bits_to_int([int(value) for tag, value in shot.entries if tag == "fourier"])
        for shot in shots.results
    ]

    observed_probabilities = np.bincount(outcomes, minlength=2**n_qubits) / n_shots

    np.testing.assert_allclose(
        observed_probabilities,
        expected_probabilities,
        rtol=0.0,
        atol=0.03,
    )
