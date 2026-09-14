"""Tests for the HHL algorithm."""

from __future__ import annotations

from typing import no_type_check

import numpy as np
import pytest
import zixy.qubit.pauli as zqp
from guppylang import comptime, guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, output
from guppylang.std.debug import state_output
from guppylang.std.quantum import discard_array, h, qubit, ry, x
from selene_sim import Quest

from guppyalgos.algorithms.linear_systems import (
    eigenvalue_inversion_angles,
    hhl,
    hhl_conditional_rotation,
    hhl_power_oracles,
)
from guppyalgos.utils import int_to_bits, qarray
from tests.helpers import assert_allclose_ignorephase, switch_endianness


@pytest.mark.parametrize(
    ("n_qpe", "time_step", "rotation_scalar", "expected_samples"),
    [
        (
            3,
            -0.5,
            1.0,
            {
                0: 0.0,
                # k=1 => lambda=1.0, ratio=1.0 => (2/pi)*arcsin(1) = 1.0
                1: 1.0,
                # k=2 => lambda=2.0, ratio=0.5 => (2/pi)*arcsin(0.5) = 1/3
                2: 1.0 / 3.0,
                # k=7 (k_signed=-1) => lambda=-1.0, ratio=-1.0 => -1.0
                7: -1.0,
            },
        ),
        (
            3,
            -0.5,
            0.5,
            {
                0: 0.0,
                1: 1.0 / 3.0,  # k=1 => lambda=1.0, ratio=0.5 => 1/3
                2: float((2.0 / np.pi) * np.arcsin(0.25)),
            },
        ),
        (
            4,
            -0.25,
            1.0,
            {
                0: 0.0,
                1: 1.0,  # k=1 => lambda=1.0, ratio=1.0 => 1.0
                2: 1.0 / 3.0,
                15: -1.0,
            },
        ),
    ],
)
def test_eigenvalue_inversion_angles(
    n_qpe: int,
    time_step: float,
    rotation_scalar: float,
    expected_samples: dict[int, float],
) -> None:
    """Test angle calculations for eigenvalue inversion across configurations."""
    angles = eigenvalue_inversion_angles(
        n_qpe=n_qpe,
        time_step=time_step,
        rotation_scalar=rotation_scalar,
    )
    assert len(angles) == 2**n_qpe
    for k, expected_val in expected_samples.items():
        assert np.isclose(angles[k], expected_val)


@pytest.mark.parametrize(
    ("ham_str", "n_qubits"),
    [
        ("(1.0, Z0)", 1),
        ("(1.5, I0), (-0.5, Z0)", 1),
        ("(1.0, X0 Y1), (0.5, Z0 Z1)", 2),
    ],
)
def test_hhl_power_oracles(ham_str: str, n_qubits: int) -> None:
    """Test that hhl_power_oracles produces callable forward and inverse oracles."""
    ham_op = zqp.RealTermSum.from_str(ham_str)
    fwd_oracle, inv_oracle = hhl_power_oracles(
        input_matrix=ham_op,
        time_step=0.25,
        n_trotter_steps=1,
        n_input_qubits=n_qubits,
    )
    assert fwd_oracle is not None
    assert inv_oracle is not None


@pytest.mark.parametrize(
    ("n_qpe", "clock_int", "angles"),
    [
        (1, 0, [0.0, 0.5]),
        (1, 1, [0.0, 0.5]),
        (2, 0, [0.0, 0.25, 0.5, 0.75]),
        (2, 1, [0.0, 0.25, 0.5, 0.75]),
        (2, 2, [0.0, 0.25, 0.5, 0.75]),
        (2, 3, [0.0, 0.25, 0.5, 0.75]),
    ],
)
def test_hhl_conditional_rotation(
    n_qpe: int, clock_int: int, angles: list[float]
) -> None:
    """Test clock-conditioned rotation for various clock register sizes and inputs."""
    rot = hhl_conditional_rotation(n_qpe, angles)
    clock_bits = int_to_bits(clock_int, n_qpe)

    if n_qpe == 1:

        @guppy
        @no_type_check
        def run_rot_1() -> None:
            clock = qarray(1)
            b0 = comptime(clock_bits[0])
            if b0:
                x(clock[0])
            ancilla = qubit()
            rot(clock, ancilla)
            state_output("final_state", ancilla)
            discard_array(clock)
            discard_array(array(ancilla))

        res = run_rot_1.emulator(n_qubits=4).run()
    else:

        @guppy
        @no_type_check
        def run_rot_2() -> None:
            clock = qarray(2)
            b0 = comptime(clock_bits[0])
            b1 = comptime(clock_bits[1])
            if b0:
                x(clock[0])
            if b1:
                x(clock[1])
            ancilla = qubit()
            rot(clock, ancilla)
            state_output("final_state", ancilla)
            discard_array(clock)
            discard_array(array(ancilla))

        res = run_rot_2.emulator(n_qubits=6).run()

    states = Quest.extract_states_dict(res.results[0].entries)
    sv = states["final_state"].get_state_vector_distribution()[0].state
    expected_theta = angles[clock_int] * np.pi / 2.0
    expected_sv = np.array(
        [np.cos(expected_theta), np.sin(expected_theta)], dtype=np.complex128
    )
    assert_allclose_ignorephase(sv, expected_sv, threshold=1e-5)


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


@pytest.mark.parametrize(
    ("ham_str", "eigen_state", "expected_eigenvalue"),
    [
        # |0> has lambda = 1.0 => ratio 1.0 => 100% success
        ("(1.5, I0), (-0.5, Z0)", 0, 1.0),
        # |1> has lambda = -1.0 => ratio -1.0 => 100% success
        ("(0.5, I0), (1.5, Z0)", 1, -1.0),
    ],
)
def test_hhl_single_shot_deterministic_eigenstate(
    ham_str: str, eigen_state: int, expected_eigenvalue: float
) -> None:
    """Test single-shot HHL on exact eigenstates with deterministic success."""
    ham_op = zqp.RealTermSum.from_str(ham_str)
    n_qpe = 3
    time_step = -0.5
    rotation_scalar = 1.0

    @guppy
    @no_type_check
    def prepare_b(qs: array[qubit, 1]) -> None:
        if comptime(eigen_state == 1):
            x(qs[0])

    hhl_op = hhl(
        input_matrix=ham_op,
        n_qpe=n_qpe,
        rotation_scalar=rotation_scalar,
        time_step=time_step,
        n_input_qubits=1,
    )

    @guppy
    @no_type_check
    def run_single_shot() -> None:
        qs = qarray(1)
        prepare_b(qs)
        success = hhl_op(qs)
        output("success", success)
        if success:
            state_output("solution", qs)
        discard_array(qs)

    sim_result = run_single_shot.emulator(n_qubits=10).run()
    assert bool(sim_result.results[0].as_dict()["success"]) is True

    states = Quest.extract_states_dict(sim_result.results[0].entries)
    actual_state = states["solution"].get_state_vector_distribution()[0].state
    expected_x = (
        np.array([1.0, 0.0], dtype=np.complex128)
        if eigen_state == 0
        else np.array([0.0, 1.0], dtype=np.complex128)
    )
    assert_allclose_ignorephase(actual_state, expected_x, threshold=1e-5)
