"""Check incrementer modifiers against independent modular arithmetic."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, control, dagger
from guppylang.std.debug import state_output
from guppylang.std.quantum import cx, discard_array, h, qubit, t
from selene_sim import QuantumReplay, Quest

from guppyalgos.primitives.arithmetic.incrementer.incrementer_cca import (
    cca_incrementer,
    required_clean_ancillas,
)
from guppyalgos.primitives.arithmetic.incrementer.linear_depth_incrementer import (
    linear_depth_incrementer,
)
from guppyalgos.testing import (
    Endianness,
    _statevector_with_zeroed_ancilla,
    assert_allclose_ignorephase,
    assert_unitary_modifiers,
)
from guppyalgos.utils import qarray
from guppyalgos.utils.guppy.array import join_arrays


@pytest.mark.parametrize("num_bits", [1, 2, 3])
@pytest.mark.parametrize("operation", [linear_depth_incrementer, cca_incrementer])
def test_incrementer_modifiers(
    operation: GuppyFunctionDefinition, num_bits: int
) -> None:
    """Check all public modes, overflow, and the one- and two-bit fast paths."""

    @guppy(unitary=True)
    @no_type_check
    def circuit(qs: array[qubit, num_bits]) -> None:
        operation(qs)

    expected = np.roll(np.eye(2**num_bits), 1, axis=0)
    assert_unitary_modifiers(
        circuit,
        num_bits,
        expected_unitary=expected,
        endianness=Endianness.LITTLE,
        n_extra_qubits=num_bits,
    )


@pytest.mark.parametrize("max_value", [0, 1, 3, 6, 15])
def test_bounded_incrementer_modifiers(max_value: int) -> None:
    """All modes preserve high bits and wrap within the selected active width."""
    num_bits = 4
    modulus = 1 << min(num_bits, (max_value + 1).bit_length())

    @guppy(unitary=True)
    @no_type_check
    def circuit(qs: array[qubit, num_bits]) -> None:
        linear_depth_incrementer(qs, max_value)

    expected = np.zeros((2**num_bits, 2**num_bits))
    for value in range(2**num_bits):
        result = value // modulus * modulus + (value + 1) % modulus
        expected[result, value] = 1
    assert_unitary_modifiers(
        circuit,
        num_bits,
        expected_unitary=expected,
        endianness=Endianness.LITTLE,
        n_extra_qubits=num_bits,
    )


@pytest.mark.parametrize("num_controls", [0, 1, 3])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize(
    ("operation", "num_bits", "active_bits", "num_ancillas"),
    [
        (linear_depth_incrementer, 5, 5, 4),
        (linear_depth_incrementer, 5, 3, 2),
        (cca_incrementer, 7, 7, required_clean_ancillas(8)),
    ],
)
def test_incrementer_coherence_and_cleanup(
    operation: GuppyFunctionDefinition,
    num_bits: int,
    active_bits: int,
    num_ancillas: int,
    inverse: bool,
    num_controls: int,
) -> None:
    """Check coherent branches and zeroed workspace across measurement outcomes.

    A phase-bearing superposition exercises full carry/borrow propagation,
    including the recursive CCA schedule and untouched bounded-register bits.
    The expected state comes directly from classical modular arithmetic.
    """
    num_qubits = num_bits + num_controls
    # Three controls require one conjunction flag and one CNX ancilla.
    num_extra = num_ancillas + (2 if num_controls > 1 else 0)

    if active_bits < num_bits:

        @guppy(unitary=True)
        @no_type_check
        def circuit(qs: array[qubit, num_bits]) -> None:
            operation(qs, 6)

    else:

        @guppy(unitary=True)
        @no_type_check
        def circuit(qs: array[qubit, num_bits]) -> None:
            operation(qs)

    # Keep empty control blocks out of the HUGR; Guppy cannot normalize them.
    if num_controls > 0:

        @guppy
        @no_type_check
        def apply(
            qs: array[qubit, num_bits], controls: array[qubit, num_controls]
        ) -> None:
            with control(controls):
                if inverse:
                    with dagger:
                        circuit(qs)
                else:
                    circuit(qs)

    else:

        @guppy
        @no_type_check
        def apply(
            qs: array[qubit, num_bits], controls: array[qubit, num_controls]
        ) -> None:
            if inverse:
                with dagger:
                    circuit(qs)
            else:
                circuit(qs)

    @guppy
    @no_type_check
    def main() -> None:
        controls = qarray(num_controls)
        qs = qarray(num_bits)
        for i in range(num_controls):
            h(controls[i])
            t(controls[i])
        h(qs[0])
        t(qs[0])
        for i in range(1, num_bits):
            cx(qs[0], qs[i])

        apply(qs, controls)

        result = join_arrays(controls, qs, num_qubits)
        state_output("result_state", result)
        discard_array(result)

    expected = np.zeros(2**num_qubits, dtype=np.complex128)
    modulus = 1 << active_bits
    delta = -1 if inverse else 1
    for controls in range(2**num_controls):
        for value in (0, 2**num_bits - 1):
            result = value
            if controls == 2**num_controls - 1:
                result = value // modulus * modulus + (value + delta) % modulus
            phase = np.exp(1j * np.pi / 4 * (controls.bit_count() + (value != 0)))
            expected[(result << num_controls) | controls] = phase / np.sqrt(
                2 ** (num_controls + 1)
            )

    # Force both outcomes of the first cleanup; sample subsequent corrections.
    replay = QuantumReplay(
        simulator=Quest(random_seed=42),
        resume_with_measurement=True,
        measurements=[[bool(i % 2)] for i in range(16)],
    )
    shots = (
        main.emulator(num_qubits + num_extra)
        .with_simulator(replay)
        .with_shots(16)
        .run()
    )
    for shot in shots.results:
        state = Quest.extract_states_dict(shot.entries)["result_state"]
        actual = _statevector_with_zeroed_ancilla(state)
        assert_allclose_ignorephase(actual, expected)
