"""Walsh-Hadamard decomposition for unitary diagonal operators.

Reference:
    Welch et al., "Efficient Quantum Circuits for Diagonal Unitaries Without
    Ancillas", arXiv:1306.3991v1 (2013).
    https://arxiv.org/abs/1306.3991

This module implements the Walsh basis decomposition of diagonal unitaries
(Eq. 7 in the reference) using the Paley-ordered circuit (Fig. 4, non-optimal).
A future optimization using Gray code / sequency ordering (Fig. 5) would reduce
gate count from O(n*2^n) to the optimal 2^(n+1)-3 two-qubit gates.
"""

from __future__ import annotations

from typing import no_type_check

import numpy as np
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import array
from guppylang.std.quantum import cx, qubit, rz


def fast_walsh_hadamard_transform(phases: np.ndarray) -> np.ndarray:
    """Unnormalized Walsh-Hadamard transform of a real phase vector.

    Applies the butterfly recursion H_{2n} = [[H_n, H_n], [H_n, -H_n]].
    Satisfies FWHT(FWHT(x)) = n * x. Input length must be a power of 2.

    Unnormalized Walsh transform convention used here::

        w_k = sum_j phases[j] * (-1)^(popcount(k & j))

    where popcount counts set bits and '&' is bitwise AND.

    Args:
        phases: Real-valued array of length 2^m for some m >= 0.

    Returns:
        Walsh coefficients of the same length.

    Raises:
        ValueError: If the input length is not a power of 2.

    """
    n = len(phases)
    if n == 0 or n & (n - 1):
        raise ValueError(f"Input length {n} is not a power of 2.")
    coeffs = np.asarray(phases, dtype=float).copy()
    block_size = 1
    while block_size < n:
        step = 2 * block_size
        for start in range(0, n, step):
            for offset in range(block_size):
                even = coeffs[start + offset]
                odd = coeffs[start + offset + block_size]
                # Butterfly update for each pair: (a, b) -> (a + b, a - b).
                coeffs[start + offset] = even + odd
                coeffs[start + offset + block_size] = even - odd
        block_size = step
    return coeffs


def diagonal_unitary_walsh(
    diagonal: np.ndarray,
    truncation_threshold: float = 1e-10,
) -> GuppyFunctionDefinition[[array[qubit, ...]], None]:
    """Synthesize a unitary diagonal operator using Walsh-Hadamard decomposition.

    Decomposes D = diag(exp(i*theta_j)) as a product of commuting Z-parity
    phase gadgets: U = prod_k exp(i*a_k * W_k), where W_k is a tensor product
    of Z on qubits selected by the set bits of k, and the a_k are computed
    via the Fast Walsh-Hadamard Transform of the phase vector.

    Little-endian convention (LSB-first):
    This implementation uses a little-endian convention: qubit qs[0] is the
    least significant bit, qs[n-1] is the most significant. Diagonal index j
    uses binary representation where bit i (0-indexed, LSB=0) is mapped to
    qubit qs[i]. This matches Endianness.LITTLE in tests (via get_unitary).

    Walsh basis viewpoint:
    each index k labels a character chi_k(j) = (-1)^(popcount(k & j)), so the
    FWHT coefficients are the phase-vector expansion in this basis.

    The truncation threshold is optional and customary in practice: keeping only
    terms with larger ``|a_k|`` yields shallower approximate circuits.

    Args:
        diagonal: 1D array of complex entries, all with magnitude 1.
            Length must be a power of 2 (2^n for n qubits).
        truncation_threshold: Walsh terms with ``|angle_rad|`` below this are
            dropped, yielding an approximate circuit. Set to 0.0 to keep all
            non-global terms.

    Returns:
        GuppyFunctionDefinition implementing the diagonal unitary on n qubits,
        where 2^n = len(diagonal). Supports dagger and control modifiers for
        the synthesized circuit. The identity Walsh coefficient is omitted,
        so controls preserve the synthesized circuit's phase convention.

    Raises:
        ValueError: If diagonal length is not a power of 2 or entries are not unitary.

    """
    diagonal = np.asarray(diagonal, dtype=complex)
    size = len(diagonal)
    if size == 0 or size & (size - 1):
        raise ValueError(f"Diagonal length {size} must be a power of 2.")
    n = size.bit_length() - 1
    if not np.allclose(np.abs(diagonal), 1.0, atol=1e-10):
        raise ValueError("Diagonal must be unitary: all entries must have magnitude 1.")
    walsh_coeffs = fast_walsh_hadamard_transform(np.angle(diagonal))

    rotations: list[float] = []
    target_masks: list[list[bool]] = []
    control_masks: list[list[bool]] = []
    for k, coeff in enumerate(walsh_coeffs):
        angle_rad = float(coeff) / size
        if abs(angle_rad) < truncation_threshold:
            continue
        # Map Walsh bit i (LSB=0) directly to qubit i (little-endian convention).
        # Example: k = 0b011 means bits i=0 and i=1 are set → qubits qs[0], qs[1].
        support = tuple(i for i in range(n) if (k >> i) & 1)
        if not support:
            continue  # global phase; no observable effect
        target = support[-1]
        controls = set(support[:-1])
        rotations.append(-2.0 * angle_rad / np.pi)
        target_masks.append([i == target for i in range(n)])
        control_masks.append([i in controls for i in range(n)])

    @guppy.comptime(unitary=True)
    @no_type_check
    def walsh_circuit(qs: array[qubit, n]) -> None:
        for theta, targets, controls in zip(
            rotations, target_masks, control_masks, strict=True
        ):
            target = targets.index(True)
            for idx in range(n):
                if controls[idx]:
                    cx(qs[idx], qs[target])
            rz(qs[target], angle(theta))
            for idx in range(n):
                if controls[idx]:
                    cx(qs[idx], qs[target])

    return walsh_circuit
