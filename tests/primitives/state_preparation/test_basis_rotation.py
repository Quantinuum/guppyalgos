"""Tests for real mode-basis state preparation."""

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, comptime
from guppylang.std.quantum import qubit, rz

from guppyalgos.primitives.rotations import givens_with_custom_rz
from guppyalgos.primitives.state_preparation import basis_rotation_implementation
from guppyalgos.testing import Endianness, get_unitary


def _occupation_unitary(mode_unitary: np.ndarray) -> np.ndarray:
    """Construct the occupation-space unitary induced by a mode matrix."""
    n_modes = mode_unitary.shape[0]
    occupation_unitary = np.zeros(
        (2**n_modes, 2**n_modes),
        dtype=np.complex128,
    )

    for input_state in range(2**n_modes):
        input_modes = [mode for mode in range(n_modes) if input_state & (1 << mode)]
        for output_state in range(2**n_modes):
            output_modes = [
                mode for mode in range(n_modes) if output_state & (1 << mode)
            ]
            if len(input_modes) != len(output_modes):
                continue
            if not input_modes:
                occupation_unitary[output_state, input_state] = 1.0
                continue
            occupation_unitary[output_state, input_state] = np.linalg.det(
                mode_unitary[np.ix_(output_modes, input_modes)]
            )

    return occupation_unitary


def _mode_rotation(theta: float) -> np.ndarray:
    """Return a two-mode mode-space rotation."""
    phase = np.pi * theta
    return np.array(
        [[np.cos(phase), np.sin(phase)], [-np.sin(phase), np.cos(phase)]],
        dtype=float,
    )


def _sample_three_mode_rotation() -> np.ndarray:
    """Return a fixed three-mode rotation with a non-trivial schedule."""
    theta0 = 0.31
    theta1 = -0.22
    return np.array(
        [
            [
                np.cos(theta0),
                -np.sin(theta0) * np.cos(theta1),
                np.sin(theta0) * np.sin(theta1),
            ],
            [
                np.sin(theta0),
                np.cos(theta0) * np.cos(theta1),
                -np.cos(theta0) * np.sin(theta1),
            ],
            [0.0, np.sin(theta1), np.cos(theta1)],
        ]
    )


@guppy
def _custom_phase(q: qubit, theta: angle) -> None:
    """Provide an injectable single-mode phase implementation."""
    rz(q, theta)


_custom_rotation = givens_with_custom_rz(_custom_phase)


def _compiled_basis_rotation(
    mode_matrix: np.ndarray,
    *,
    use_custom_elementaries: bool = False,
) -> tuple[np.ndarray, list[tuple[int, int, float]], list[tuple[int, float]], float]:
    """Compile a generated basis rotation and return it with its schedules."""
    rotation = _custom_rotation if use_custom_elementaries else None
    phase = _custom_phase if use_custom_elementaries else rz
    basis_rotation, rotations, phases, global_phase = basis_rotation_implementation(
        mode_matrix,
        elementary_rotation=rotation,
        elementary_phase_rz_method=phase,
    )
    n_modes = mode_matrix.shape[0]

    @guppy
    @no_type_check
    def main(qreg: array[qubit, comptime(n_modes)]) -> None:
        basis_rotation(qreg)

    return (
        get_unitary(main, n_modes, endianness=Endianness.LITTLE),
        rotations,
        phases,
        global_phase,
    )


@pytest.mark.parametrize(
    ("mode_matrix", "use_custom_elementaries"),
    [
        pytest.param(
            np.diag([-1.0, 1.0]),
            False,
            id="signed-two-mode-diagonal",
        ),
        pytest.param(
            _sample_three_mode_rotation(),
            False,
            id="three-mode-rotation",
        ),
        pytest.param(
            np.diag([-1.0, 1.0]) @ _mode_rotation(0.17),
            True,
            id="custom-elementaries",
        ),
    ],
)
def test_basis_rotation_matches_occupation_unitary(
    mode_matrix: np.ndarray,
    use_custom_elementaries: bool,
) -> None:
    """Check generated state-preparation circuits and injected elementaries."""
    compiled, rotations, phases, global_phase = _compiled_basis_rotation(
        mode_matrix,
        use_custom_elementaries=use_custom_elementaries,
    )

    assert all(q1 == q0 + 1 for q0, q1, _ in rotations)
    expected = _occupation_unitary(mode_matrix)
    np.testing.assert_allclose(
        np.exp(1j * global_phase) * compiled,
        expected,
        atol=1e-8,
    )
    assert all(abs(phase) > 0.0 for _, phase in phases)


def test_basis_rotation_compiles_empty_schedule() -> None:
    """Check that an identity basis rotation still produces a Guppy callable."""
    compiled, rotations, phases, global_phase = _compiled_basis_rotation(np.eye(1))

    np.testing.assert_allclose(compiled, np.eye(2), atol=1e-8)
    assert rotations == []
    assert phases == []
    assert global_phase == 0.0


def test_basis_rotation_does_not_prune_near_pi_rotation() -> None:
    """Check that near-π pruning follows ``atol`` without relative tolerance."""
    epsilon = 1e-5
    mode_matrix = np.array(
        [
            [-np.cos(epsilon), -np.sin(epsilon)],
            [np.sin(epsilon), -np.cos(epsilon)],
        ]
    )

    _, rotations, _, _ = basis_rotation_implementation(mode_matrix)

    assert len(rotations) == 1
    assert abs(abs(rotations[0][2]) - np.pi) > 1e-10


@pytest.mark.parametrize(
    "mode_matrix",
    [
        pytest.param(np.ones((2, 3)), id="non-square"),
        pytest.param(np.array([[1.0, 1.0], [0.0, 1.0]]), id="non-orthogonal"),
        pytest.param(
            np.array([[1.0 + 1.0j, 0.0], [0.0, 1.0]], dtype=np.complex128),
            id="complex",
        ),
        pytest.param(np.array([[np.nan]]), id="nan"),
        pytest.param(np.array([[np.inf]]), id="infinite"),
        pytest.param(np.eye(0), id="empty"),
    ],
)
def test_basis_rotation_rejects_invalid_input(mode_matrix: np.ndarray) -> None:
    """Check that invalid mode matrices are rejected at the Python boundary."""
    with pytest.raises(ValueError, match=r"matrix|mode_matrix"):
        basis_rotation_implementation(mode_matrix)


@pytest.mark.parametrize("atol", [-1.0, np.nan, np.inf])
def test_basis_rotation_rejects_invalid_tolerance(atol: float) -> None:
    """Check that schedule tolerances are finite and non-negative."""
    with pytest.raises(ValueError, match="atol"):
        basis_rotation_implementation(np.eye(2), atol=atol)
