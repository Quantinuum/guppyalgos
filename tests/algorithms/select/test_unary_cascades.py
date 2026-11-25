"""Tests for the unary cascade modifier API."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import comptime, guppy
from guppylang.std.builtins import array, control, dagger
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, h, qubit, s, toffoli, z
from selene_sim import Quest

from guppyalgos.algorithms.select import (
    adjacent_and,
    compute_cascade,
    uncompute_cascade,
)
from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and,
    temp_and_uncompute,
)
from guppyalgos.testing import (
    Endianness,
    assert_allclose_ignorephase,
    assert_unitary_modifiers,
)
from guppyalgos.utils import qarray


@pytest.mark.parametrize("start_depth", [0, 1])
def test_compute_cascade_modifiers(start_depth: int) -> None:
    """Compare the full and partial cascades with a Boolean permutation."""

    @guppy(unitary=True)
    @no_type_check
    def apply(index: array[qubit, 3], work: array[qubit, 2]) -> None:
        compute_cascade(toffoli, 2, start_depth, array(True, False, False), index, work)

    @guppy.comptime(unitary=True)
    @no_type_check
    def circuit(qs: array[qubit, 5]) -> None:
        apply(qs[:3], qs[3:])

    expected = np.zeros((32, 32), dtype=np.complex128)
    for column in range(32):
        address = column & 0b111
        low, high = (column >> 3) & 1, (column >> 4) & 1
        if start_depth == 0:
            high ^= int(address < 2)
        low ^= high & (address & 1)
        expected[address | (low << 3) | (high << 4), column] = 1

    assert_unitary_modifiers(
        circuit, 5, expected_unitary=expected, endianness=Endianness.LITTLE
    )


@pytest.mark.parametrize(("depth", "top_bit"), [(1, False), (1, True), (2, False)])
def test_adjacent_and_modifiers(depth: int, top_bit: bool) -> None:
    """Check the two prefix sources and both index-control polarities."""

    @guppy.comptime(unitary=True)
    @no_type_check
    def circuit(qs: array[qubit, 5]) -> None:
        adjacent_and(depth, qs[:3], qs[3:], [False, True, top_bit])

    expected = np.zeros((32, 32), dtype=np.complex128)
    for column in range(32):
        if depth == 1:
            flip = ((column >> 2) & 1) == top_bit
            target = 4
        else:
            flip = (column >> 4) & 1
            target = 3
        expected[column ^ (int(flip) << target), column] = 1

    assert_unitary_modifiers(
        circuit, 5, expected_unitary=expected, endianness=Endianness.LITTLE
    )


@pytest.mark.parametrize("controlled", [False, True])
def test_temporary_and_cascade_transition(controlled: bool) -> None:
    """Check phase kickback through an adjacent transition and clean workspace."""

    @guppy
    @no_type_check
    def circuit() -> None:
        index = qarray(3)
        controls = qarray(2)
        work = qarray(2)
        first_bits = array(True, False, False)
        second_bits = array(False, True, False)
        for i in range(3):
            h(index[i])
        for i in range(2):
            h(controls[i])
        s(controls[0])

        if comptime(controlled):
            with control(controls):
                compute_cascade(temp_and, 2, 0, first_bits, index, work)
            z(work[0])
            with control(controls), dagger:
                compute_cascade(temp_and, 2, 1, first_bits, index, work)
            with control(controls):
                adjacent_and(1, index, work, second_bits)
                compute_cascade(temp_and, 2, 1, second_bits, index, work)
            s(work[0])
            with control(controls), dagger:
                compute_cascade(temp_and, 2, 0, second_bits, index, work)
        else:
            compute_cascade(temp_and, 2, 0, first_bits, index, work)
            z(work[0])
            uncompute_cascade(1, 2, first_bits, index, work, temp_and_uncompute)
            adjacent_and(1, index, work, second_bits)
            compute_cascade(temp_and, 2, 1, second_bits, index, work)
            s(work[0])
            with dagger:
                compute_cascade(temp_and, 2, 0, second_bits, index, work)

        state_output("index", index)
        state_output("controls", controls)
        state_output("work", work)
        discard_array(index)
        discard_array(controls)
        discard_array(work)

    emulator = circuit.emulator(9 if controlled else 7).with_seed(42).with_shots(16)
    expected = np.zeros((8, 4, 4), dtype=np.complex128)
    expected[:, :, 0] = 1 / np.sqrt(32)
    expected[:, 2:, 0] *= 1j
    active = 3 if controlled else slice(None)
    # The snapshot puts qubit 0 first, so little-endian addresses 1 and 2
    # occupy rows 4 and 2 of the index axis.
    expected[4, active, 0] *= -1
    expected[2, active, 0] *= 1j

    for shot in emulator.run().results:
        states = Quest.extract_states_dict(shot.entries)
        state = states["index"]
        state.specified_qubits = [
            *states["index"].specified_qubits,
            *states["controls"].specified_qubits,
            *states["work"].specified_qubits,
        ]
        assert_allclose_ignorephase(state.get_single_state(), expected.flatten())
