"""Tests for QROM-composed two-qubit Givens rotations."""

from collections.abc import Callable
from math import log2
from typing import no_type_check

import numpy as np
import pytest
from guppylang import comptime, guppy
from guppylang.std.builtins import array
from guppylang.std.debug import state_result
from guppylang.std.quantum import discard_array, h, qubit, x
from selene_quest_plugin.state import SeleneQuestState
from selene_sim import Quest

from guppyalgos.algorithms.select.qrom import qrom_unary_iteration
from guppyalgos.primitives.rotations import (
    GivensRotationPhaseGradient,
    QROMRotations,
    GivensRotationRegisterIncremented,
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


def _expected_qrom_givens_state(theta: float) -> np.ndarray:
    """Return the expected state from bits ``[False, True]`` in little-endian order."""
    phase = np.pi * theta
    state = np.zeros(4, dtype=np.complex128)
    state[bits_to_int([True, False])] = -np.sin(phase)
    state[bits_to_int([False, True])] = np.cos(phase)
    return state


def _little_endian_basis_state(bits: list[bool]) -> np.ndarray:
    """Return the basis state indexed by the little-endian bit string ``bits``."""
    state = np.zeros(2 ** len(bits), dtype=np.complex128)
    state[bits_to_int(bits)] = 1.0
    return state


def _expected_theta(selected_bits: list[bool]) -> float:
    """Return the shared Givens angle encoded by the selected QROM bits."""
    encoded_integer = bits_to_int(selected_bits)
    return float(2.0 * encoded_integer / (2 ** len(selected_bits)))


def _increment_table(n_phase_qubits: int) -> list[list[bool]]:
    """Return the full increment table for a `2**n_phase_qubits`-entry QROM."""
    n_index_elements = 2**n_phase_qubits
    return [int_to_bits(index, n_phase_qubits) for index in range(n_index_elements)]


def _run_register_increment_qrom_givens_rotation(
    data_input: list[list[bool]],
) -> dict[str, SeleneQuestState]:
    """Run QROM rotations in superposition with register-incremented Givens."""
    n_data_qubits = len(data_input[0])
    n_index_qubits = int(log2(len(data_input)))
    qrom_compute = qrom_unary_iteration(data_input)
    qrom_uncompute = qrom_unary_iteration(data_input)

    @guppy
    @no_type_check
    def main() -> None:
        index_qreg = qarray(comptime(n_index_qubits))
        data_qreg = qarray(comptime(n_data_qubits))
        rot_regs = (qubit(), qubit())
        rotation = GivensRotationRegisterIncremented[comptime(n_data_qubits)]()

        x(rot_regs[1])
        qrom_rot = QROMRotations(
            qrom_compute[array[qubit, comptime(n_data_qubits)]],
            rotation,
            qrom_uncompute[array[qubit, comptime(n_data_qubits)]],
        )

        transversal(h, index_qreg)

        qrom_rot.compose(index_qreg, data_qreg, rot_regs)

        state_result("index", index_qreg)
        state_result("givens", rot_regs[0], rot_regs[1])
        state_result("qrom_target", data_qreg)
        discard_array(index_qreg)
        discard_array(data_qreg)
        rot_regs[0].discard()
        rot_regs[1].discard()

    total_qubits = n_data_qubits + (2 * n_index_qubits) + 2
    res = main.emulator(n_qubits=total_qubits).with_seed(42).run()
    return Quest.extract_states_dict(res.results[0].entries)


def _run_phase_gradient_qrom_givens_rotation(
    data_input: list[list[bool]],
) -> dict[str, SeleneQuestState]:
    """Run QROM rotations in superposition with phase-gradient Givens synthesis."""
    n_data_qubits = len(data_input[0])
    n_index_qubits = int(log2(len(data_input)))
    qrom_compute = qrom_unary_iteration(data_input)
    qrom_uncompute = qrom_unary_iteration(data_input)
    prepare_phase_gradient = phase_gradient(
        n_data_qubits,
        convention=Convention.Standard,
    )

    @guppy
    @no_type_check
    def main() -> None:
        index_qreg = qarray(comptime(n_index_qubits))
        data_qreg = qarray(comptime(n_data_qubits))
        phase_gradient_state = qarray(comptime(n_data_qubits))
        rot_regs = (qubit(), qubit())
        prepare_phase_gradient(phase_gradient_state)
        rotation = GivensRotationPhaseGradient(phase_gradient_state)

        x(rot_regs[1])
        qrom_rot = QROMRotations(
            qrom_compute[array[qubit, comptime(n_data_qubits)]],
            rotation,
            qrom_uncompute[array[qubit, comptime(n_data_qubits)]],
        )

        transversal(h, index_qreg)

        qrom_rot.compose(index_qreg, data_qreg, rot_regs)

        state_result("index", index_qreg)
        state_result("givens", rot_regs[0], rot_regs[1])
        state_result("qrom_target", data_qreg)
        discard_array(index_qreg)
        discard_array(data_qreg)
        discard_array(qrom_rot.rotation_box.phase_gradient)
        rot_regs[0].discard()
        rot_regs[1].discard()

    total_qubits = n_index_qubits + (4 * n_data_qubits) + 2
    res = main.emulator(n_qubits=total_qubits).with_seed(42).run()
    return Quest.extract_states_dict(res.results[0].entries)


@pytest.mark.parametrize(
    ("n_phase_qubits"),
    [3, 4],
    ids=["8_indices", "16_indices"],
)
@pytest.mark.parametrize(
    "run_qrom_rotation",
    [
        pytest.param(
            _run_register_increment_qrom_givens_rotation,
            id="register_increment_givens",
        ),
        pytest.param(
            _run_phase_gradient_qrom_givens_rotation,
            id="phase_gradient_givens",
        ),
    ],
)
def test_qrom_givens_rotation_superposition(
    n_phase_qubits: int,
    run_qrom_rotation: Callable[[list[list[bool]]], dict[str, SeleneQuestState]],
) -> None:
    """Check every QROM-selected Givens branch from one superposition execution."""
    data_input = _increment_table(n_phase_qubits)
    n_index_qubits = int(log2(len(data_input)))
    expected_zero = np.zeros(2**n_phase_qubits, dtype=np.complex128)
    expected_zero[0] = 1.0

    index_bit_strings_le = [
        int_to_bits(index, n_index_qubits) for index in range(len(data_input))
    ]

    states = run_qrom_rotation(data_input)
    projected_givens_states = extract_state_branches_in_superposition(
        states,
        "index",
        ["givens"],
        index_bit_strings_le,
    )
    qrom_target_state = states["qrom_target"].get_single_state()

    assert_allclose_ignorephase(qrom_target_state, expected_zero)

    zero_rotation_state = projected_givens_states[tuple(index_bit_strings_le[0])]
    expected_input_state = _little_endian_basis_state([False, True])
    assert_allclose_ignorephase(
        zero_rotation_state.state.state,
        expected_input_state,
    )

    for index, selected_bits in enumerate(data_input):
        projected_givens = projected_givens_states[tuple(index_bit_strings_le[index])]
        expected_theta = _expected_theta(selected_bits)
        expected_state = _expected_qrom_givens_state(expected_theta)
        assert_allclose_ignorephase(
            projected_givens.state.state,
            expected_state,
        )


def test_phase_gradient_qrom_givens_rotation_uncompute() -> None:
    """Check a QROM-backed phase-gradient rotation followed by its inverse.

    A fixed index selects one angle for the forward Givens rotation and the same
    angle for its inverse. The pair must restore the rotation targets while each
    QROM operation uncomputes its data register.

    """
    n_data_qubits = 3
    n_index_qubits = 3
    data_input = _increment_table(n_data_qubits)
    qrom_compute = qrom_unary_iteration(data_input)
    qrom_uncompute = qrom_unary_iteration(data_input)
    prepare_phase_gradient = phase_gradient(
        n_data_qubits,
        convention=Convention.Standard,
    )
    index_bits = int_to_bits(5, n_index_qubits)

    @guppy
    @no_type_check
    def main() -> None:
        index_qreg = qarray(comptime(n_index_qubits))
        data_qreg = qarray(comptime(n_data_qubits))
        phase_gradient_state = qarray(comptime(n_data_qubits))
        rot_regs = (qubit(), qubit())
        prepare_phase_gradient(phase_gradient_state)
        rotation = GivensRotationPhaseGradient(phase_gradient_state)

        bits = comptime(index_bits)
        for bit in range(comptime(n_index_qubits)):
            if bits[bit]:
                x(index_qreg[bit])
        x(rot_regs[1])

        qrom_rot = QROMRotations(
            qrom_compute[array[qubit, comptime(n_data_qubits)]],
            rotation,
            qrom_uncompute[array[qubit, comptime(n_data_qubits)]],
        )
        qrom_rot.compose(index_qreg, data_qreg, rot_regs)
        qrom_rot.daggered(index_qreg, data_qreg, rot_regs)

        state_result("givens", rot_regs[0], rot_regs[1])
        state_result("qrom_target", data_qreg)
        discard_array(index_qreg)
        discard_array(data_qreg)
        discard_array(qrom_rot.rotation_box.phase_gradient)
        rot_regs[0].discard()
        rot_regs[1].discard()

    res = main.emulator(n_qubits=14).with_seed(42).run()
    states = Quest.extract_states_dict(res.results[0].entries)
    expected_zero = np.zeros(2**n_data_qubits, dtype=np.complex128)
    expected_zero[0] = 1.0
    expected_givens = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)

    assert_allclose_ignorephase(states["qrom_target"].get_single_state(), expected_zero)
    assert_allclose_ignorephase(states["givens"].get_single_state(), expected_givens)
