"""Tests for QROM-backed Givens rotation cascades."""

from math import cos, pi, sin
from typing import no_type_check

import numpy as np
import pytest
from guppylang import comptime, guppy
from guppylang.std.builtins import array
from guppylang.std.debug import state_result
from guppylang.std.quantum import discard_array, h, qubit, x
from selene_sim import Quest

from guppyalgos.algorithms.select.qrom import qrom_unary_iteration
from guppyalgos.primitives.rotations import (
    GivensCascadePhaseGradient,
    QROMRotations,
)
from guppyalgos.primitives.state_preparation.phase_gradient import (
    Convention,
    phase_gradient,
)
from guppyalgos.utils import (
    discard_nested_arrays,
    bits_to_int,
    int_to_bits,
    qarray,
    transversal,
)
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    extract_state_branches_in_superposition,
)


def _theta(bits: list[bool]) -> float:
    """Return the Givens angle encoded by a little-endian fixed-point word."""
    encoded_integer = bits_to_int(bits)
    return float(2.0 * encoded_integer / (2 ** len(bits)))


def _apply_expected_givens(
    state: np.ndarray,
    n_modes: int,
    pair_index: int,
    theta: float,
) -> np.ndarray:
    """Apply a neighboring-mode Givens rotation to an expected statevector.

    For every fixed configuration of the other modes, the rotation acts on the
    two-dimensional subspace spanned by ``|10>`` and ``|01>`` for modes
    ``pair_index`` and ``pair_index + 1``. In that ordered basis it applies

    ``[[cos(pi * theta), -sin(pi * theta)],``
    `` [sin(pi * theta),  cos(pi * theta)]]``.

    The ``|00>`` and ``|11>`` amplitudes are unchanged. The loop visits only the
    ``|10>`` member of each pair and finds its ``|01>``
    partner by swapping the two mode bits; this updates every coupled pair exactly
    once. Amplitudes are always read from ``state`` and written to a copy so that
    one update cannot affect a later calculation.

    Args:
        state: Input statevector of length ``2**n_modes``.
        n_modes: Number of qubit modes represented by ``state``.
        pair_index: Index of the first mode in the neighboring pair.
        theta: Rotation angle in half-turn units; the angle in radians is
            ``pi * theta``.

    Returns:
        A rotated copy of ``state``.

    """
    phase = pi * theta
    c = cos(phase)
    s = sin(phase)
    next_state = state.copy()

    for basis in range(2**n_modes):
        bits = int_to_bits(basis, n_modes)
        if not bits[pair_index] or bits[pair_index + 1]:
            continue

        swapped_bits = bits.copy()
        swapped_bits[pair_index] = False
        swapped_bits[pair_index + 1] = True
        swapped_basis = bits_to_int(swapped_bits)

        amp_10 = state[basis]
        amp_01 = state[swapped_basis]
        next_state[basis] = (c * amp_10) - (s * amp_01)
        next_state[swapped_basis] = (s * amp_10) + (c * amp_01)

    return next_state


def _expected_cascade_state(
    selected_angle_bits: list[list[bool]],
    n_modes: int,
) -> np.ndarray:
    """Return the expected state after the nearest-neighbor Givens cascade."""
    state = np.zeros(2**n_modes, dtype=np.complex128)
    state[1] = 1.0
    for pair_index, bits in enumerate(selected_angle_bits):
        state = _apply_expected_givens(state, n_modes, pair_index, _theta(bits))
    return state


def test_phase_gradient_qrom_givens_cascade_two_rotations() -> None:
    """Check two QROM-selected rotations in a phase-gradient Givens cascade.

    Each QROM entry contains two angle words. The cascade uses the first word to
    rotate modes 0 and 1, then the second to rotate modes 1 and 2, reusing one
    phase-gradient state for both rotations. The index is prepared in uniform
    superposition so every QROM entry is checked in one emulator run. The initial
    excitation is placed on mode 0, ensuring that the first rotation feeds
    amplitude into the second rotation. Applying the inverse cascade must then
    restore that input state. Both QROM operations also uncompute the two angle
    registers to zero.

    """
    n_index_qubits = 2
    n_data_qubits = 2
    n_givens = 2
    n_modes = 3
    data_input = [
        [[False, False], [False, False]],
        [[True, False], [False, True]],
        [[False, True], [True, False]],
        [[True, True], [True, True]],
    ]
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
        data_qregs = array(
            qarray(comptime(n_data_qubits)) for _ in range(comptime(n_givens))
        )
        rotation_regs = qarray(comptime(n_modes))
        pg_state = qarray(n_data_qubits)

        prepare_phase_gradient(pg_state)
        cascade = GivensCascadePhaseGradient[
            n_data_qubits,
            n_givens,
            n_modes,
        ](pg_state)

        transversal(h, index_qreg)

        x(rotation_regs[0])
        qrom_rot = QROMRotations(
            qrom_compute[array[array[qubit, n_data_qubits], n_givens]],
            cascade,
            qrom_uncompute,
        )
        qrom_rot.compose(index_qreg, data_qregs, rotation_regs)

        state_result("index_after_forward", index_qreg)
        state_result("modes_after_forward", rotation_regs)
        state_result("qrom_target_0_after_forward", data_qregs[0])
        state_result("qrom_target_1_after_forward", data_qregs[1])

        qrom_rot.daggered(index_qreg, data_qregs, rotation_regs)

        state_result("index_after_inverse", index_qreg)
        state_result("modes_after_inverse", rotation_regs)
        state_result("qrom_target_0_after_inverse", data_qregs[0])
        state_result("qrom_target_1_after_inverse", data_qregs[1])
        discard_array(index_qreg)
        discard_nested_arrays(data_qregs)
        discard_array(rotation_regs)
        discard_array(qrom_rot.rotation_box.phase_gradient)

    res = main.emulator(n_qubits=15).with_seed(42).run()
    states = Quest.extract_states_dict(res.results[0].entries)
    expected_qrom_zero = np.zeros(2**n_data_qubits, dtype=np.complex128)
    expected_qrom_zero[0] = 1.0

    for stage in ("forward", "inverse"):
        assert_allclose_ignorephase(
            states[f"qrom_target_0_after_{stage}"].get_single_state(),
            expected_qrom_zero,
        )
        assert_allclose_ignorephase(
            states[f"qrom_target_1_after_{stage}"].get_single_state(),
            expected_qrom_zero,
        )

    selected_indices = range(len(data_input))
    selected_bit_strings = [
        int_to_bits(selected_index, n_index_qubits)
        for selected_index in selected_indices
    ]
    projected_modes = extract_state_branches_in_superposition(
        states,
        "index_after_forward",
        ["modes_after_forward"],
        selected_bit_strings,
    )

    for selected_index, selected_bits in zip(
        selected_indices, selected_bit_strings, strict=True
    ):
        expected_modes = _expected_cascade_state(
            data_input[selected_index],
            n_modes,
        )
        assert_allclose_ignorephase(
            projected_modes[tuple(selected_bits)].state.state,
            expected_modes,
        )

    projected_inverse_modes = extract_state_branches_in_superposition(
        states,
        "index_after_inverse",
        ["modes_after_inverse"],
        selected_bit_strings,
    )
    expected_input_state = np.zeros(2**n_modes, dtype=np.complex128)
    expected_input_state[1] = 1.0
    for selected_bits in selected_bit_strings:
        assert_allclose_ignorephase(
            projected_inverse_modes[tuple(selected_bits)].state.state,
            expected_input_state,
        )


def test_phase_gradient_givens_cascade_rejects_too_many_rotations() -> None:
    """Reject a cascade that has more rotations than neighboring mode pairs."""
    prepare_phase_gradient = phase_gradient(
        2,
        convention=Convention.Standard,
    )

    @guppy
    @no_type_check
    def main() -> None:
        rotation_regs = qarray(3)
        pg_state = qarray(2)
        prepare_phase_gradient(pg_state)
        data_qregs = array(qarray(2) for _ in range(3))
        cascade = GivensCascadePhaseGradient[2, 3, 3](pg_state)
        cascade.compose(data_qregs, rotation_regs)
        discard_nested_arrays(data_qregs)
        discard_array(rotation_regs)
        discard_array(cascade.phase_gradient)

    with pytest.raises(ValueError, match="requires n_givens < n_modes"):
        main.compile()
