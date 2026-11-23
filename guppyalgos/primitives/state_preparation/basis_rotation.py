"""Prepare a state with a real mode-basis rotation."""

from __future__ import annotations

from typing import no_type_check

import numpy
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle, pi
from guppylang.std.builtins import Unitary, array, comptime
from guppylang.std.quantum import qubit, rz
from numpy.typing import NDArray


def basis_rotation_implementation(
    mode_matrix: NDArray[numpy.float64],
    elementary_rotation: GuppyFunctionDefinition[[qubit, qubit, angle], None]
    | Unitary[[qubit, qubit, angle], None]
    | None = None,
    elementary_phase_rz_method: GuppyFunctionDefinition[[qubit, angle], None] = rz,
    atol: float = 1e-10,
) -> tuple[
    GuppyFunctionDefinition,
    list[tuple[int, int, float]],
    list[tuple[int, float]],
    float,
]:
    """Build a Guppy callable for a real mode-basis rotation.

    ``mode_matrix`` is an ``n x n`` real orthogonal matrix acting on modes or
    orbitals, not a ``2**n x 2**n`` Hilbert-space unitary. A basis rotation is
    the mode/orbital transformation and its induced occupation-space unitary.

    The decomposition uses nearest-neighbor Givens elimination in a QR-style
    scheme. It emits at most ``n * (n - 1) / 2`` Givens rotations, up to ``n``
    single-qubit phase corrections, and a global phase. The schedule is
    emitted serially; parallel circuit depth is not optimized.

    The matrix has no angle units. NumPy computes schedule angles and the
    global phase in radians via ``arctan2`` and ``numpy.angle``. At the Guppy
    boundary, these values are converted to half-turns because Guppy's
    ``angle`` values use half-turns. Consequently, injected Guppy rotation and
    phase functions receive half-turns, while the returned schedules use
    radians; the same conceptual angles therefore use two units across this
    boundary. The generated circuit implements the induced occupation-space
    unitary up to ``exp(-1j * global_phase)``.

    Returns the callable, Givens schedule, single-qubit phase schedule, and
    global phase. ``elementary_phase_rz_method`` handles the phase schedule.
    The default Givens rotation uses the standard ``rz`` implementation; a
    supplied ``elementary_rotation`` may use a different RZ implementation.

    References:
        I. D. Kivlichan et al., "Quantum Simulation of Electronic Structure
        with Linear Depth and Connectivity," Phys. Rev. Lett. 120, 110501
        (2018). https://doi.org/10.1103/PhysRevLett.120.110501

    """
    if not numpy.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and non-negative")

    raw_matrix = numpy.asarray(mode_matrix, dtype=numpy.complex128)
    if raw_matrix.ndim != 2 or 0 in raw_matrix.shape:
        raise ValueError(
            f"matrix must be a non-empty two-dimensional array, got {raw_matrix.shape}"
        )
    if not numpy.isfinite(raw_matrix).all():
        raise ValueError("matrix must contain only finite values")
    if numpy.linalg.norm(raw_matrix.imag) > atol * raw_matrix.shape[0]:
        raise ValueError("matrix must be real-valued")

    working_matrix = raw_matrix.real.copy()

    if working_matrix.shape[0] != working_matrix.shape[1]:
        raise ValueError(f"mode_matrix must be square, got {working_matrix.shape}")

    n_qubits = working_matrix.shape[0]

    if (
        numpy.linalg.norm(working_matrix.T @ working_matrix - numpy.eye(n_qubits))
        > atol * n_qubits
    ):
        raise ValueError("mode_matrix must be orthogonal")

    scheduled_rotations: list[tuple[int, int, float]] = []
    for layer in range(max(0, 2 * n_qubits - 3)):
        for row1, column in (
            (n_qubits - 1 - layer + 2 * column, column)
            for column in range(
                max(0, layer - n_qubits + 2),
                layer // 2 + 1,
            )
        ):
            row0 = row1 - 1
            theta = numpy.arctan2(
                -working_matrix[row1, column], working_matrix[row0, column]
            )
            if numpy.isclose(
                numpy.absolute(theta),
                numpy.pi,
                atol=atol,
                rtol=0.0,
            ):
                theta = 0.0
            if numpy.isclose(theta, 0.0, atol=atol, rtol=0.0):
                continue

            cos_theta = numpy.cos(theta)
            sin_theta = numpy.sin(theta)
            vals0 = working_matrix[row0].copy()
            vals1 = working_matrix[row1].copy()
            working_matrix[row0] = cos_theta * vals0 - sin_theta * vals1
            working_matrix[row1] = sin_theta * vals0 + cos_theta * vals1
            scheduled_rotations.append((row0, row1, float(theta)))

    scheduled_rotations.reverse()
    diagonal_phases = [float(numpy.angle(value)) for value in working_matrix.diagonal()]
    scheduled_phases = [
        (q, phase)
        for q, phase in enumerate(diagonal_phases)
        if not numpy.isclose(phase, 0.0, atol=atol, rtol=0.0)
    ]
    global_phase = float(sum(diagonal_phases) / 2.0)

    if elementary_rotation is None:
        from guppyalgos.primitives.rotations.givens_rotation import givens_rotation

        elementary_rotation = givens_rotation

    # Typed placeholders keep empty comptime schedules valid; flags skip their gates.
    phase_schedule = scheduled_phases or [(0, 0.0)]
    rotation_schedule = scheduled_rotations or [(0, 1, 0.0)]
    has_phases = bool(scheduled_phases)
    has_rotations = bool(scheduled_rotations)

    @guppy
    @no_type_check
    def basis_rotation(qreg: array[qubit, comptime(n_qubits)]) -> None:
        """Apply the precomputed basis-rotation schedules to ``qreg``."""
        for q, phi in comptime(phase_schedule):
            if comptime(has_phases):
                elementary_phase_rz_method(qreg[q], phi / comptime(numpy.pi) * pi)

        for q0, q1, theta in comptime(rotation_schedule):
            if comptime(has_rotations):
                elementary_rotation(
                    qreg[q0],
                    qreg[q1],
                    theta / comptime(numpy.pi) * pi,
                )

    return basis_rotation, scheduled_rotations, scheduled_phases, global_phase
