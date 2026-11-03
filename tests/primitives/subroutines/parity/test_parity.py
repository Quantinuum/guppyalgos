"""Tests for sequential and LAQCC parity circuits."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.builtins import array, control, dagger, nat, output
from guppylang.std.quantum import collect_measurements, measure, measure_array, qubit, x

from guppyalgos.primitives.subroutines.parity import (
    parity_laqcc,
    parity_laqcc_total_qubits,
    parity_sequential,
)
from guppyalgos.utils import qarray
from guppyalgos.testing import assert_unitary_modifiers
from guppyalgos.utils.guppy.unsafe_borrow import (
    _unsafe_array_borrow_slice,
    _unsafe_array_unborrow_slice,
)


N_INPUT_QUBITS = 5


def test_parity_sequential_modifiers() -> None:
    """Check sequential parity and its modifiers against the exact permutation."""

    @guppy.unitary
    class circuit:
        @guppy
        @no_type_check
        def __call__(qs: array[qubit, 3]) -> None:
            """Borrow the data register and apply the selected modifier."""
            prefix, inputs, suffix = _unsafe_array_borrow_slice(qs, 1, 2, 0)
            parity_sequential(prefix[0], inputs)
            _unsafe_array_unborrow_slice(qs, prefix, inputs, suffix)

        @guppy
        @no_type_check
        def daggered(qs: array[qubit, 3]) -> None:
            """Borrow the data register and apply the selected modifier."""
            prefix, inputs, suffix = _unsafe_array_borrow_slice(qs, 1, 2, 0)
            with dagger:
                parity_sequential(prefix[0], inputs)
            _unsafe_array_unborrow_slice(qs, prefix, inputs, suffix)

        @guppy
        @no_type_check
        def controlled[k: nat](qs: array[qubit, 3], cs: array[qubit, k]) -> None:
            """Borrow the data register and apply the selected modifier."""
            prefix, inputs, suffix = _unsafe_array_borrow_slice(qs, 1, 2, 0)
            with control(cs):
                parity_sequential(prefix[0], inputs)
            _unsafe_array_unborrow_slice(qs, prefix, inputs, suffix)

        @guppy
        @no_type_check
        def ctrl_daggered[k: nat](qs: array[qubit, 3], cs: array[qubit, k]) -> None:
            """Borrow the data register and apply the selected modifier."""
            prefix, inputs, suffix = _unsafe_array_borrow_slice(qs, 1, 2, 0)
            with control(cs), dagger:
                parity_sequential(prefix[0], inputs)
            _unsafe_array_unborrow_slice(qs, prefix, inputs, suffix)

    # In |target, input0, input1> order, odd input parity flips the target.
    expected = np.eye(8, dtype=np.complex128)[:, [0, 5, 6, 3, 4, 1, 2, 7]]
    assert_unitary_modifiers(circuit, 3, expected_unitary=expected)


@pytest.mark.parametrize(
    ("n_input_qubits", "expected_total"),
    [
        pytest.param(3, 4, id="sequential-fallback"),
        pytest.param(8, 19, id="eight-inputs"),
        pytest.param(16, 43, id="sixteen-inputs"),
    ],
)
def test_parity_laqcc_total_qubits(n_input_qubits: int, expected_total: int) -> None:
    """Count inputs, target, and LAQCC ancillas."""
    assert parity_laqcc_total_qubits(n_input_qubits) == expected_total


def _get_parity_emulator(parity, total_qubits: int):
    """Build an emulator for a parity implementation."""

    @guppy
    @no_type_check
    def main(bits: array[bool, N_INPUT_QUBITS], target_bit: bool) -> None:
        inputs = qarray(N_INPUT_QUBITS)
        target = qubit()

        for i in range(N_INPUT_QUBITS):
            if bits[i]:
                x(inputs[i])
        if target_bit:
            x(target)

        parity(target, inputs)

        output("inputs", collect_measurements(measure_array(inputs)))
        output("target", measure(target).read())

    return main.emulator(n_qubits=total_qubits).with_seed(42).with_shots(1)


@pytest.mark.parametrize(
    ("parity", "total_qubits"),
    [
        pytest.param(
            parity_sequential,
            N_INPUT_QUBITS + 1,
            id="sequential",
        ),
        pytest.param(
            parity_laqcc,
            parity_laqcc_total_qubits(N_INPUT_QUBITS),
            id="laqcc",
        ),
    ],
)
@pytest.mark.parametrize(
    ("input_bits", "target_bit", "expected_target"),
    [
        pytest.param(
            [False] * N_INPUT_QUBITS,
            False,
            False,
            id="even-input-target-zero",
        ),
        pytest.param(
            [True, *([False] * (N_INPUT_QUBITS - 1))],
            True,
            False,
            id="odd-input-target-one",
        ),
        pytest.param(
            [True, *([False] * (N_INPUT_QUBITS - 1))],
            False,
            True,
            id="odd-input-target-zero",
        ),
        pytest.param(
            [True, True, *([False] * (N_INPUT_QUBITS - 2))],
            True,
            True,
            id="even-input-target-one",
        ),
    ],
)
def test_parity_basis_states(
    parity,
    total_qubits: int,
    input_bits: list[bool],
    target_bit: bool,
    expected_target: bool,
) -> None:
    """XOR input parity into the target without changing the inputs."""
    emulator = _get_parity_emulator(parity, total_qubits)
    shots = emulator.run(bits=input_bits, target_bit=target_bit).collated_shots()

    assert shots[0]["inputs"][0] == input_bits
    assert shots[0]["target"][0] == expected_target
