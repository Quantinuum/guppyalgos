"""Tests for the HHL algorithm."""

from __future__ import annotations

from typing import no_type_check

import numpy as np
import pytest
import zixy.qubit.pauli as zqp
from guppylang import comptime, guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, h, qubit, ry, x
from selene_sim import Quest

from guppyalgos.algorithms.linear_systems import hhl
from guppyalgos.utils import qarray
from tests.helpers import assert_allclose_ignorephase, switch_endianness


@pytest.mark.parametrize(
    (
        "ham_str",
        "n_input_qubits",
        "prep_kind",
        "prep_param",
        "n_qpe",
        "time_step",
        "rotation_scalar",
    ),
    [
        # 1-qubit case 1: Diagonal system A = diag(1, 2), b = cos(t)|0> + sin(t)|1>
        ("(1.5, I0), (-0.5, Z0)", 1, "ry", 0.35, 3, -0.5, 1.0),
        # 1-qubit case 2: Diagonal system with smaller rotation scalar
        ("(1.5, I0), (-0.5, Z0)", 1, "ry", 0.5, 3, -0.5, 0.5),
        # 1-qubit case 3: Non-diagonal system A = [[1.5, 0.5], [0.5, 1.5]], b = |1>
        ("(1.5, I0), (0.5, X0)", 1, "x", 0.0, 3, -0.5, 1.0),
        # 1-qubit case 4: Non-diagonal system, b = |+>
        ("(1.5, I0), (0.5, X0)", 1, "h", 0.0, 3, -0.5, 1.0),
        # 1-qubit case 5: Signed eigenvalues A = diag(2.0, -1.0), b = |+>
        ("(0.5, I0), (1.5, Z0)", 1, "h", 0.0, 3, -0.5, 1.0),
        # 1-qubit case 6: Signed eigenvalues, b = cos(theta)|0> + sin(theta)|1>
        ("(0.5, I0), (1.5, Z0)", 1, "ry", 0.4, 3, -0.5, 1.0),
        # 2-qubit case 1: 2-qubit diagonal system, b = |++>
        ("(1.5, I0 I1), (-0.5, Z0 I1)", 2, "hh", 0.0, 3, -0.5, 1.0),
        # 2-qubit case 2: 2-qubit non-diagonal system, b = |01> (x on qs[1])
        ("(1.5, I0 I1), (0.5, X0 I1)", 2, "x1", 0.0, 3, -0.5, 1.0),
        # 2-qubit case 3: 2-qubit non-diagonal system, b = |10> (x on qs[0])
        ("(1.5, I0 I1), (0.5, X0 I1)", 2, "x0", 0.0, 3, -0.5, 1.0),
        # 2-qubit case 4: 2-qubit coupled system, b = |+0> (h on qs[0])
        ("(1.5, I0 I1), (-0.5, Z0 Z1)", 2, "h0", 0.0, 3, -0.5, 1.0),
        # 2-qubit case 5: 2-qubit coupled system, b = |0+> (h on qs[1])
        ("(1.5, I0 I1), (-0.5, Z0 Z1)", 2, "h1", 0.0, 3, -0.5, 1.0),
        # 3-qubit case 1: 3-qubit diagonal system, b = |+++>
        ("(1.5, I0 I1 I2), (-0.5, Z0 I1 I2)", 3, "hhh", 0.0, 3, -0.5, 1.0),
        # 3-qubit case 2: 3-qubit coupled non-diagonal system, b = |+00>
        ("(1.5, I0 I1 I2), (0.5, X0 Z1 Z2)", 3, "h0", 0.0, 3, -0.5, 1.0),
        # 3-qubit case 3: 3-qubit signed eigenvalues, b = |+++>
        ("(0.5, I0 I1 I2), (1.5, Z0 Z1 Z2)", 3, "hhh", 0.0, 3, -0.5, 1.0),
    ],
)
def test_hhl_rus(
    ham_str: str,
    n_input_qubits: int,
    prep_kind: str,
    prep_param: float,
    n_qpe: int,
    time_step: float,
    rotation_scalar: float,
) -> None:
    """Test HHL across 1-, 2-, and 3-qubit linear systems with repeat-until-success."""
    ham_op = zqp.RealTermSum.from_str(ham_str)

    n_sim_qubits = n_qpe + n_input_qubits + 3

    if n_input_qubits == 1:
        if prep_kind == "ry":

            @guppy
            @no_type_check
            def prepare_b_1(qs: array[qubit, 1]) -> None:
                ry(qs[0], angle(comptime(prep_param)))

            theta = prep_param * np.pi / 2.0
            b_vec = np.array([np.cos(theta), np.sin(theta)], dtype=np.complex128)
        elif prep_kind == "x":

            @guppy
            @no_type_check
            def prepare_b_1(qs: array[qubit, 1]) -> None:
                x(qs[0])

            b_vec = np.array([0.0, 1.0], dtype=np.complex128)
        else:

            @guppy
            @no_type_check
            def prepare_b_1(qs: array[qubit, 1]) -> None:
                h(qs[0])

            b_vec = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)

        hhl_op_1 = hhl(
            input_matrix=ham_op,
            n_qpe=n_qpe,
            rotation_scalar=rotation_scalar,
            time_step=time_step,
            n_input_qubits=1,
        )

        @guppy
        @no_type_check
        def run_hhl_rus_1() -> None:
            while True:
                qs = qarray(1)
                prepare_b_1(qs)
                success = hhl_op_1(qs)
                if success:
                    state_output("solution", qs)
                    discard_array(qs)
                    break
                discard_array(qs)

        sim_result = run_hhl_rus_1.emulator(n_qubits=n_sim_qubits).run()

    elif n_input_qubits == 2:
        if prep_kind == "hh":

            @guppy
            @no_type_check
            def prepare_b_2(qs: array[qubit, 2]) -> None:
                h(qs[0])
                h(qs[1])

            b_vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex128)
        elif prep_kind == "x1":

            @guppy
            @no_type_check
            def prepare_b_2(qs: array[qubit, 2]) -> None:
                x(qs[1])

            b_vec = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128)
        elif prep_kind == "x0":

            @guppy
            @no_type_check
            def prepare_b_2(qs: array[qubit, 2]) -> None:
                x(qs[0])

            b_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)
        elif prep_kind == "h0":

            @guppy
            @no_type_check
            def prepare_b_2(qs: array[qubit, 2]) -> None:
                h(qs[0])

            b_vec = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.complex128) / np.sqrt(2)
        else:

            @guppy
            @no_type_check
            def prepare_b_2(qs: array[qubit, 2]) -> None:
                h(qs[1])

            b_vec = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.complex128) / np.sqrt(2)

        hhl_op_2 = hhl(
            input_matrix=ham_op,
            n_qpe=n_qpe,
            rotation_scalar=rotation_scalar,
            time_step=time_step,
            n_input_qubits=2,
        )

        @guppy
        @no_type_check
        def run_hhl_rus_2() -> None:
            while True:
                qs = qarray(2)
                prepare_b_2(qs)
                success = hhl_op_2(qs)
                if success:
                    state_output("solution", qs)
                    discard_array(qs)
                    break
                discard_array(qs)

        sim_result = run_hhl_rus_2.emulator(n_qubits=n_sim_qubits).run()

    else:
        if prep_kind == "hhh":

            @guppy
            @no_type_check
            def prepare_b_3(qs: array[qubit, 3]) -> None:
                h(qs[0])
                h(qs[1])
                h(qs[2])

            b_vec = np.full(8, 1.0 / np.sqrt(8), dtype=np.complex128)
        elif prep_kind == "h0":

            @guppy
            @no_type_check
            def prepare_b_3(qs: array[qubit, 3]) -> None:
                h(qs[0])

            b_vec = np.zeros(8, dtype=np.complex128)
            b_vec[0] = 1.0 / np.sqrt(2)
            b_vec[1] = 1.0 / np.sqrt(2)
        else:

            @guppy
            @no_type_check
            def prepare_b_3(qs: array[qubit, 3]) -> None:
                x(qs[0])

            b_vec = np.zeros(8, dtype=np.complex128)
            b_vec[1] = 1.0

        hhl_op_3 = hhl(
            input_matrix=ham_op,
            n_qpe=n_qpe,
            rotation_scalar=rotation_scalar,
            time_step=time_step,
            n_input_qubits=3,
        )

        @guppy
        @no_type_check
        def run_hhl_rus_3() -> None:
            while True:
                qs = qarray(3)
                prepare_b_3(qs)
                success = hhl_op_3(qs)
                if success:
                    state_output("solution", qs)
                    discard_array(qs)
                    break
                discard_array(qs)

        sim_result = run_hhl_rus_3.emulator(n_qubits=n_sim_qubits).run()

    states = Quest.extract_states_dict(sim_result.results[0].entries)
    actual_state = switch_endianness(
        states["solution"].get_state_vector_distribution()[0].state
    )

    a_mat = ham_op.to_sparse_matrix(False).toarray()
    expected_x = np.linalg.solve(a_mat, b_vec)
    expected_x /= np.linalg.norm(expected_x)

    assert_allclose_ignorephase(actual_state, expected_x, threshold=1e-5)
