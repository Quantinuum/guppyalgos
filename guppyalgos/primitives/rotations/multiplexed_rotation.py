"""Ancilla-free uniformly controlled single-qubit rotations."""

from __future__ import annotations

from typing import no_type_check

from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, frozenarray, nat
from guppylang.std.quantum import cx, qubit, rz

from .rotation_helper import RotationAxis


@guppy
@no_type_check
def multiplexed_rotation[n_controls: nat, n_angles: nat, Axis: RotationAxis](
    axis: Axis,
    angles: frozenarray[float, n_angles],
    controls: array[qubit, n_controls],
    target: qubit,
) -> None:
    r"""Apply an ancilla-free uniformly controlled rotation.

    For ``angles[k]`` in half-turns, this applies

    .. math::

        \sum_k |k\rangle\langle k|_{\mathrm{controls}} \otimes
        R_{\mathrm{axis}}(\pi\,\mathrm{angles}[k])_{\mathrm{target}}.

    The controls are little-endian. ``angles`` must contain one entry for each
    control-register basis state, i.e. ``n_angles == 2**n_controls``.
    The angle transform uses an in-place fast Walsh-Hadamard transform.

    Args:
        axis: Basis-change object selecting the X, Y, or Z rotation axis.
        angles: One half-turn rotation angle for every control-register basis state.
        controls: Little-endian control register.
        target: Qubit receiving the controlled rotation.

    """
    dim = 2**n_controls
    coefficients = array(angles[index] for index in range(n_angles))
    block_size = 1
    while block_size < dim:
        step = 2 * block_size
        start = 0
        while start < dim:
            offset = 0
            while offset < block_size:
                even = coefficients[start + offset]
                odd = coefficients[start + offset + block_size]
                coefficients[start + offset] = even + odd
                coefficients[start + offset + block_size] = even - odd
                offset += 1
            start += step
        block_size = step

    axis.prepare_basis(target)
    for gray_index in range(dim):
        gray_code = gray_index ^ (gray_index >> 1)
        coefficient_index = 0
        for bit in range(n_controls):
            coefficient_index = (coefficient_index << 1) | ((gray_code >> bit) & 1)
        rz(target, angle(coefficients[coefficient_index] / float(dim)))

        if n_controls > 0:
            toggled_bit = 0
            counter = gray_index + 1
            while counter % 2 == 0 and toggled_bit < n_controls - 1:
                counter //= 2
                toggled_bit += 1
            cx(controls[n_controls - 1 - toggled_bit], target)
    axis.restore_basis(target)
