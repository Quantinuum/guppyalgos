"""Multiplexor-based state preparation implemented directly in Guppy."""

from collections.abc import Sequence
from enum import IntEnum
from math import ceil, log2, pi
from typing import no_type_check

import numpy as np
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import array, nat
from guppylang.std.quantum import cx, qubit, ry, rz

# Treat smaller amplitudes and rotation angles as numerical zero.
_EPS = 1e-11


class _CommandType(IntEnum):
    ROTATION = 0
    CX = 1


class _RecursionNodeType(IntEnum):
    LEFT = 0
    RIGHT = 1
    ROOT = 2


def _demultiplex_rotation(
    angles: list[float],
    total_qubits: int,
    node_type: _RecursionNodeType,
) -> list[tuple[_CommandType, float | int]]:
    """Decompose a uniformly controlled rotation into rotations and CX gates."""
    midpoint = len(angles) // 2
    p_angles = [(angles[i] - angles[midpoint + i]) / 2 for i in range(midpoint)]
    q_angles = [(angles[i] + angles[midpoint + i]) / 2 for i in range(midpoint)]
    if node_type == _RecursionNodeType.RIGHT:
        p_angles, q_angles = q_angles, p_angles

    commands: list[tuple[_CommandType, float | int]] = []
    if len(q_angles) == 1:
        commands.append((_CommandType.ROTATION, q_angles[0]))
    else:
        commands.extend(
            _demultiplex_rotation(q_angles, total_qubits, _RecursionNodeType.LEFT)
        )

    control = total_qubits - int(log2(len(angles))) - 1
    commands.append((_CommandType.CX, control))

    if len(p_angles) == 1:
        commands.append((_CommandType.ROTATION, p_angles[0]))
    else:
        commands.extend(
            _demultiplex_rotation(p_angles, total_qubits, _RecursionNodeType.RIGHT)
        )

    if node_type == _RecursionNodeType.ROOT:
        commands.append((_CommandType.CX, control))
    return commands


def _multiplexed_rotation_commands(
    angles: list[float],
) -> list[tuple[_CommandType, float | int]]:
    """Return Gray-code decomposition for a multiplexed rotation."""
    if not any(abs(value) > _EPS for value in angles):
        return []
    if len(angles) == 1:
        return [(_CommandType.ROTATION, angles[0])]
    return _demultiplex_rotation(
        angles, int(log2(len(angles))) + 1, _RecursionNodeType.ROOT
    )


def _state_preparation_angles(
    state: np.ndarray,
) -> tuple[np.ndarray, list[float], list[float]]:
    """Compute one recursive layer's angles and reduced statevector."""
    half_length = len(state) // 2
    radii = np.zeros(half_length)
    phases = np.zeros(half_length)
    y_angles = np.zeros(half_length)
    z_angles = np.zeros(half_length)

    for index in range(half_length):
        a = state[2 * index]
        b = state[2 * index + 1]
        if abs(a) < _EPS and abs(b) < _EPS:
            continue

        norm = abs(a) ** 2 + abs(b) ** 2
        radius = 1.0 if abs(norm - 1.0) <= _EPS else float(np.sqrt(norm))
        a /= radius
        b /= radius

        theta_a = float(np.angle(a) / pi)
        theta_b = float(np.angle(b) / pi)
        magnitude_a = float(np.clip(abs(a), 0.0, 1.0))
        y_angle = 2 * np.arccos(magnitude_a) / pi
        z_angle = theta_b - theta_a
        if abs(y_angle) > _EPS:
            y_angles[index] = y_angle
        if abs(z_angle) > _EPS:
            z_angles[index] = z_angle
        radii[index] = radius
        phases[index] = theta_a + 0.5 * z_angle

    reduced_state = radii * np.exp(1j * pi * phases)
    return reduced_state, y_angles.tolist(), z_angles.tolist()


def _state_preparation_layers(
    state: np.ndarray,
) -> list[
    tuple[
        int,
        list[tuple[_CommandType, float | int]],
        list[tuple[_CommandType, float | int]],
    ]
]:
    """Compute the multiplexed rotations in circuit application order."""
    n_qubits = int(log2(len(state)))
    psi = state
    layers: list[
        tuple[
            int,
            list[tuple[_CommandType, float | int]],
            list[tuple[_CommandType, float | int]],
        ]
    ] = []

    for step in range(n_qubits):
        psi, y_angles, z_angles = _state_preparation_angles(psi)

        target = n_qubits - step - 1
        layers.append(
            (
                target,
                _multiplexed_rotation_commands(y_angles),
                _multiplexed_rotation_commands(z_angles),
            )
        )

    return list(reversed(layers))


def multiplexor_prep[n: nat](
    state: Sequence[complex] | np.ndarray,
) -> GuppyFunctionDefinition[[array[qubit, n]], None]:
    r"""Create a Guppy function that prepares an arbitrary normalized state.

    The input amplitudes must satisfy :math:`\sum_i |\alpha_i|^2 = 1`. They use
    little-endian basis indexing, so qubit ``q[0]`` is the least-significant bit
    of the amplitude index. The amplitudes are preserved exactly and padded
    with zeros to the next power-of-two length.

    The implementation recursively partitions the amplitudes and applies
    multiplexed Ry rotations to set their magnitudes and multiplexed Rz
    rotations to set their phases. Each multiplexor is emitted directly as
    Guppy ``Ry``/``Rz`` and ``CX`` operations using a Gray-code decomposition.
    The resulting function prepares

    .. math::

        \sum_i \alpha_i |i\rangle

    on an input qubit array initialized to :math:`|0\rangle`.

    See Appendix B.2 of `Hamiltonian dynamics simulation using linear combination
    of unitaries on an ion trap quantum computer
    <https://arxiv.org/abs/2501.18515>`_ for the recursive multiplexed-rotation
    decomposition.

    Args:
        state: One-dimensional normalized statevector. Its length need not be a
            power of two.

    Returns:
        A Guppy function that prepares the zero-padded state on its input qubits.

    Raises:
        ValueError: If ``state`` is not one-dimensional, is empty, or is not
            normalized.

    """
    state_array = np.asarray(state, dtype=np.complex128)
    if state_array.ndim != 1 or state_array.size == 0:
        raise ValueError("State must be a non-empty one-dimensional statevector.")
    if not np.isclose(np.linalg.norm(state_array), 1.0):
        raise ValueError("Statevector must be normalized.")

    n_qubits = ceil(log2(len(state_array)))
    padded_state = np.pad(state_array, (0, 2**n_qubits - len(state_array)))
    # Reorder the public little-endian amplitudes for the recursive pairing order.
    synthesis_state = padded_state.reshape((2,) * n_qubits).transpose().reshape(-1)
    layers = _state_preparation_layers(synthesis_state)

    @guppy.comptime(unitary=True)
    @no_type_check
    def multiplexor_prep_fn(qs: array[qubit, n_qubits]) -> None:
        for target, y_commands, z_commands in layers:
            for command, value in y_commands:
                if command == _CommandType.ROTATION:
                    ry(qs[target], angle(value))
                else:
                    cx(qs[value], qs[target])
            for command, value in z_commands:
                if command == _CommandType.ROTATION:
                    rz(qs[target], angle(value))
                else:
                    cx(qs[value], qs[target])

    return multiplexor_prep_fn
