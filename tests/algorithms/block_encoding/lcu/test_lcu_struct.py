"""Tests for the generic LCU register interface."""

from typing import no_type_check

import numpy as np
from guppylang import guppy
from guppylang.std.builtins import array, bool
from guppylang.std.debug import state_output
from guppylang.std.quantum import cx, discard_array, qubit, x

from guppyalgos.algorithms.block_encoding.lcu.lcu import LCU
from guppyalgos.utils import apply_bitstring, int_to_bits, qarray
from guppyalgos.testing import get_statevector_projected


@guppy.struct
class PrepRegs:
    """PREPARE registers used by the generic LCU struct."""

    index1: array[qubit, 2]
    index2: array[qubit, 2]


@guppy.struct
class TargetRegs:
    """Target registers used by the generic LCU struct."""

    targets: array[qubit, 2]


def test_lcu_accepts_register_structs() -> None:
    """Test the matrix of an LCU whose PREPARE and target registers are structs.

    The circuit is evaluated on every two-qubit computational basis state of the
    target register. Its PREPARE operation sets one control in each preparation
    subregister, so SELECT
    applies an X gate to both target qubits. UNPREPARE returns all preparation
    qubits to zero, allowing them to be projected out before the resulting target
    statevectors are assembled into the expected X tensor X matrix.

    """

    @guppy
    @no_type_check
    def prepare(prep: PrepRegs) -> None:
        x(prep.index1[0])
        x(prep.index2[0])

    @guppy
    @no_type_check
    def select(prep: PrepRegs, targets: TargetRegs) -> None:
        cx(prep.index1[0], targets.targets[0])
        cx(prep.index2[0], targets.targets[1])

    @guppy
    @no_type_check
    def unprepare(prep: PrepRegs) -> None:
        x(prep.index1[0])
        x(prep.index2[0])

    @guppy
    @no_type_check
    def main(bits: array[bool, 2]) -> None:
        prep = PrepRegs(qarray(2), qarray(2))
        targets = TargetRegs(qarray(2))
        apply_bitstring(targets.targets, bits)
        lcu = LCU(prepare, select, unprepare)
        lcu.compose(prep, targets)
        state_output("index1", prep.index1)
        state_output("index2", prep.index2)
        discard_array(prep.index1)
        discard_array(prep.index2)
        discard_array(targets.targets)

    emulator = main.emulator(6)
    actual = np.column_stack(
        [
            get_statevector_projected(
                emulator,
                6,
                {"index1": [False] * 2, "index2": [False] * 2},
                main_args_dict={"bits": int_to_bits(i, 2)},
            )
            for i in range(4)
        ]
    )
    expected = np.fliplr(np.eye(4))
    assert np.allclose(np.abs(actual), expected)
