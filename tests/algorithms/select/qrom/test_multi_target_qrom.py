"""Tests for multi-target unary-iteration QROM."""

from typing import no_type_check

import numpy as np
from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, x
from selene_sim import Quest

from guppyalgos.primitives.measurement import discard_nested_array
from guppyalgos.algorithms.select.qrom import qrom_unary_iteration
from guppyalgos.utils import int_to_bits, qarray
from guppyalgos.testing import assert_allclose_ignorephase


def test_qrom_unary_iteration_two_target_registers() -> None:
    """Check QROM fanout into two target registers."""
    n_index_qubits = 2
    n_state_qubits = 2
    n_target_registers = 2
    data_input = [
        [[False, False], [False, False]],
        [[True, False], [False, True]],
        [[False, True], [True, False]],
        [[True, True], [True, True]],
    ]
    idxs = list(range(4))
    qrom = qrom_unary_iteration(data_input)

    @guppy
    @no_type_check
    def main(bits: array[bool, n_state_qubits]) -> None:
        index_qreg = qarray(n_index_qubits)
        target_qregs = array(qarray(n_state_qubits) for _ in range(n_target_registers))

        for bit in range(n_index_qubits):
            if bits[bit]:
                x(index_qreg[bit])

        qrom(index_qreg, target_qregs)

        state_output(
            "target",
            target_qregs[0][0],
            target_qregs[0][1],
            target_qregs[1][0],
            target_qregs[1][1],
        )
        discard_array(index_qreg)
        discard_nested_array(target_qregs)

    emulator = main.emulator(n_qubits=7)

    for index in idxs:
        index_bits = int_to_bits(index, n_index_qubits)
        res = emulator.run(bits=index_bits)
        target_state = Quest.extract_states_dict(res.results[0].entries)[
            "target"
        ].get_single_state()
        target_bits = data_input[index][0] + data_input[index][1]
        expected_state = np.zeros(2 ** (n_state_qubits * n_target_registers))
        expected_state[int("".join("1" if bit else "0" for bit in target_bits), 2)] = (
            1.0
        )
        assert_allclose_ignorephase(target_state, expected_state)
