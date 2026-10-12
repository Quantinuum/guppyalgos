"""Tests for the composable QROM rotation struct."""

from collections.abc import Callable
from math import ceil, log2
from typing import Any, no_type_check

import numpy as np
import pytest
from guppylang import comptime, guppy
from guppylang.std.builtins import array
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard, discard_array, h, qubit
from selene_sim import Quest

from guppyalgos.algorithms.select.qrom import qrom_unary_iteration
from guppyalgos.primitives.rotations import (
    RotationPhaseGradient,
    QROMRotations,
    RotationAxisZ,
    RotationRegisterIncremented,
)
from guppyalgos.primitives.state_preparation.phase_gradient import (
    Convention,
    phase_gradient,
)
from guppyalgos.utils import (
    bits_to_int,
    int_to_bits,
    qarray,
    transversal,
)
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    extract_state_branches_in_superposition,
)


def _expected_rz_plus_state(theta: float) -> np.ndarray:
    """Return ``Rz(theta)|+>`` in tket's half-turn convention."""
    phase = np.pi * theta / 2.0
    return np.array(
        [np.exp(-1j * phase), np.exp(1j * phase)],
        dtype=np.complex128,
    ) / np.sqrt(2.0)


def _expected_theta(selected_bits: list[bool]) -> float:
    """Return the shared kickback angle for both rotation constructions."""
    encoded_integer = bits_to_int(selected_bits)
    return float(2.0 * encoded_integer / (2 ** len(selected_bits)))


def _increment_table(n_phase_qubits: int) -> list[list[bool]]:
    """Return the full increment table for a `2**n_phase_qubits`-entry QROM."""
    n_index_elements = 2**n_phase_qubits
    return [int_to_bits(index, n_phase_qubits) for index in range(n_index_elements)]


def _run_register_increment_qrom_rotation(
    data_input: list[list[bool]],
) -> dict[str, Any]:
    """Run QROM rotations in superposition with register-increment synthesis."""
    n_data_qubits = len(data_input[0])
    n_index_qubits = ceil(log2(len(data_input)))
    qrom_compute = qrom_unary_iteration(data_input)
    qrom_uncompute = qrom_unary_iteration(data_input)

    @guppy
    @no_type_check
    def main() -> None:
        index_qreg = qarray(comptime(n_index_qubits))
        rotation_target = qubit()
        data_qreg = qarray(comptime(n_data_qubits))
        rotation = RotationRegisterIncremented[comptime(n_data_qubits), RotationAxisZ](
            RotationAxisZ()
        )

        transversal(h, index_qreg)
        h(rotation_target)

        qrom_rot = QROMRotations(
            qrom_compute[array[qubit, comptime(n_data_qubits)]],
            rotation,
            qrom_uncompute[array[qubit, comptime(n_data_qubits)]],
        )

        qrom_rot.compose(index_qreg, data_qreg, rotation_target)

        state_output("index", index_qreg)
        state_output("ancilla", rotation_target)
        state_output("qrom_target", data_qreg)
        discard_array(index_qreg)
        discard_array(data_qreg)
        discard(rotation_target)

    total_qubits = n_data_qubits + (2 * n_index_qubits) + 1
    res = main.emulator(n_qubits=total_qubits).with_seed(42).run()
    return Quest.extract_states_dict(res.results[0].entries)


def _run_phase_gradient_qrom_rotation(
    data_input: list[list[bool]],
) -> dict[str, Any]:
    """Run QROM rotations in superposition with phase-gradient synthesis."""
    n_data_qubits = len(data_input[0])
    n_index_qubits = ceil(log2(len(data_input)))
    qrom_compute = qrom_unary_iteration(data_input)
    qrom_uncompute = qrom_unary_iteration(data_input)
    fourier_state = phase_gradient(
        n_data_qubits,
        convention=Convention.Standard,
    )

    @guppy
    @no_type_check
    def main() -> None:
        index_qreg = qarray(comptime(n_index_qubits))
        rotation_target = qubit()
        data_qreg = qarray(comptime(n_data_qubits))
        phase_gradient_state = qarray(comptime(n_data_qubits))
        fourier_state(phase_gradient_state)
        rotation = RotationPhaseGradient(
            phase_gradient_state,
            RotationAxisZ(),
        )

        transversal(h, index_qreg)
        h(rotation_target)

        qrom_rot = QROMRotations(
            qrom_compute[array[qubit, comptime(n_data_qubits)]],
            rotation,
            qrom_uncompute[array[qubit, comptime(n_data_qubits)]],
        )

        qrom_rot.compose(index_qreg, data_qreg, rotation_target)

        state_output("index", index_qreg)
        state_output("ancilla", rotation_target)
        state_output("qrom_target", data_qreg)
        discard_array(index_qreg)
        discard_array(data_qreg)
        discard_array(qrom_rot.rotation_box.phase_gradient)
        discard(rotation_target)

    total_qubits = n_index_qubits + (3 * n_data_qubits) + 1
    res = main.emulator(n_qubits=total_qubits).with_seed(42).run()
    return Quest.extract_states_dict(res.results[0].entries)


@pytest.mark.parametrize(
    ("n_phase_qubits"),
    [3, 4, 5],
    ids=["8_indices", "16_indices", "32_indices"],
)
@pytest.mark.parametrize(
    "run_qrom_rotation",
    [
        pytest.param(_run_register_increment_qrom_rotation, id="register_increment"),
        pytest.param(_run_phase_gradient_qrom_rotation, id="phase_gradient"),
    ],
)
def test_qrom_rotation_superposition(
    n_phase_qubits: int,
    run_qrom_rotation: Callable[[list[list[bool]]], dict[str, Any]],
) -> None:
    """Check every QROM-selected ``Rz`` branch from one superposition execution.

    The test prepares a uniform superposition over all QROM indices, applies the
    generic QROM rotation gadget once, and then projects the final state onto
    each computational-basis index branch. For every branch, it verifies that:

    1. the ancilla matches the expected ``Rz|+>`` state for the table entry
       stored at that index, and
    2. the QROM-loaded data register has been fully uncomputed back to ``|0...0>``.

    This checks both supported rotation syntheses against the same QROM table:
    register-incremented rotations and phase-gradient kickback rotations.

    """
    data_input = _increment_table(n_phase_qubits)
    n_index_qubits = int(log2(len(data_input)))
    expected_zero = np.zeros(2**n_phase_qubits, dtype=np.complex128)
    expected_zero[0] = 1.0

    index_bitstrings_le = [
        int_to_bits(index, n_index_qubits) for index in range(len(data_input))
    ]

    states = run_qrom_rotation(data_input)
    projected_ancilla_states = extract_state_branches_in_superposition(
        states,
        "index",
        ["ancilla"],
        index_bitstrings_le,
    )
    qrom_target_state = states["qrom_target"].get_single_state()

    # QROM compute -> rotate -> uncompute should return the data register to zero.
    assert_allclose_ignorephase(qrom_target_state, expected_zero)

    for index, selected_bits in enumerate(data_input):
        # Look up the projected ancilla state for this index branch and compare it
        # with the expected one-qubit ``Rz|+>`` state from the encoded table entry.
        projected_ancilla = projected_ancilla_states[tuple(index_bitstrings_le[index])]
        expected_theta = _expected_theta(selected_bits)
        expected_state = _expected_rz_plus_state(expected_theta)
        assert_allclose_ignorephase(projected_ancilla.state.state, expected_state)
