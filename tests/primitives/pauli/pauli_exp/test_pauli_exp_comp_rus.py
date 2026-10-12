"""Pauli Exponential tests."""

from __future__ import annotations

from guppyalgos.utils import qarray
from guppylang import guppy

from guppylang.std.builtins import comptime
from guppylang.std.angles import angle
from guppylang.std.quantum import h, discard_array
import pytest

from guppyalgos.primitives.subroutines.ladders import CXLadderLog, Ladder

import zixy.qubit.pauli as zqp


from guppyalgos.primitives.pauli.pauli_exp import pauli_exp

from scipy.linalg import expm
import numpy as np
from guppyalgos.testing import assert_allclose_ignorephase
from guppyalgos.primitives.rotations import (
    comparator_based_rz_cascade,
    n_comparator_based_rz_cascade_ancillas,
)
from guppyalgos.utils import transversal
from guppylang.std.debug import state_result
from selene_sim import Quest


from typing import no_type_check


def pauli_exp_test_fn(
    paulis: zqp.String, n_state_qubits: int, cx_ladder: type[Ladder]
) -> None:
    """Test the pauli exponential comparator rz implementation."""
    epsilon = 0.001
    rus_rz = comparator_based_rz_cascade(epsilon)

    pauli_g = pauli_exp(paulis, n_state_qubits, cx_ladder, rus_rz)

    theta = 0.7

    @guppy
    @no_type_check
    def main() -> None:
        state_qreg = qarray(comptime(n_state_qubits))
        transversal(h, state_qreg)
        pauli_g(state_qreg, angle(comptime(theta)))
        state_result("result_state", state_qreg)
        discard_array(state_qreg)

    n_rz_ancillas = n_comparator_based_rz_cascade_ancillas(epsilon)
    n_sim_qubits = n_state_qubits + n_rz_ancillas

    em_result = main.emulator(n_sim_qubits).with_seed(1).run()
    states = Quest.extract_states_dict(em_result.results[0])
    guppy_sv = states["result_state"].get_single_state()

    # TOO slow
    # guppy_sv = get_statevector(main, n_state_qubits+n_rus_qubits)

    pauli_mat = paulis.to_sparse_matrix(True).todense()
    u_mat = expm(-1j * (0.5 * np.pi * (theta)) * pauli_mat)
    h_state = (1 / np.sqrt(2**n_state_qubits)) * np.ones(2**n_state_qubits)
    numpy_sv = u_mat @ h_state

    assert_allclose_ignorephase(numpy_sv, guppy_sv, threshold=0.01)


@pytest.mark.parametrize(
    ("p_str", "n_state_qubits", "cx_ladder"),
    [
        (pauli_str, n_qubits, cx_method)
        for pauli_str in [
            "X0 Y1",
            "Y0 Z1",
            "Z0 X1",
            "X0 X1",
        ]
        for n_qubits in [2]
        for cx_method in [CXLadderLog]
    ],
)
def test_pauli_exp_2q(
    p_str: str,
    n_state_qubits: int,
    cx_ladder: type[Ladder],
) -> None:
    """Test 2-qubit pauli exponentials over various pauli strings and methods."""
    paulis = zqp.String.from_str(p_str, n_state_qubits)
    pauli_exp_test_fn(paulis, n_state_qubits, cx_ladder)
