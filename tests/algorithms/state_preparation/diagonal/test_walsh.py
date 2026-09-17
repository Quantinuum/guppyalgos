"""Tests for Walsh-Hadamard diagonal unitary synthesis."""

from __future__ import annotations

import numpy as np
import pytest

from guppyalgos.algorithms.state_preparation.diagonal import (
    diagonal_unitary_walsh,
    fast_walsh_hadamard_transform,
)
from guppyalgos.testing import Endianness, assert_allclose_ignorephase, get_unitary


# ---------------------------------------------------------------------------
# Classical FWHT tests (no Guppy, fast)
# ---------------------------------------------------------------------------


def test_fwht_known_2_element() -> None:
    """FWHT of [a, b] should give [a+b, a-b]."""
    result = fast_walsh_hadamard_transform(np.array([1.0, 2.0]))
    np.testing.assert_allclose(result, [3.0, -1.0])


def test_fwht_roundtrip() -> None:
    """Applying FWHT twice should yield n * original (self-inverse up to scale)."""
    phases = np.array([0.1, 0.5, -0.3, 0.8])
    roundtrip = fast_walsh_hadamard_transform(fast_walsh_hadamard_transform(phases))
    np.testing.assert_allclose(roundtrip, len(phases) * phases)


def test_fwht_invalid_length() -> None:
    """FWHT should raise ValueError for non-power-of-2 input length."""
    with pytest.raises(ValueError, match="power of 2"):
        fast_walsh_hadamard_transform(np.array([1.0, 2.0, 3.0]))


def test_fwht_cz_diagonal_sanity_check() -> None:
    """Sanity check: FWHT coefficients for CZ diagonal [1, 1, 1, -1].

    In little-endian basis convention (least significant qubit first), the
    diagonal [1, 1, 1, -1] represents the CZ gate matrix's diagonal on 2 qubits
    where index 3 (binary 11) receives the phase flip when both qubits are |1>.

    The unnormalized FWHT of phases [0, 0, 0, pi] should yield [pi, -pi, -pi, pi],
    which when divided by 4 gives normalized coefficients [pi/4, -pi/4, -pi/4, pi/4].
    These correspond to Walsh parity terms at indices k=0,1,2,3.
    """
    # Phases extracted from diagonal: 0 at indices 0,1,2 (where diagonal[j]=1)
    # and pi at index 3 (where diagonal[j]=-1, i.e. exp(i*pi) = -1)
    phases = np.array([0.0, 0.0, 0.0, np.pi])

    # Unnormalized FWHT should be [pi, -pi, -pi, pi]
    coeffs = fast_walsh_hadamard_transform(phases)
    np.testing.assert_allclose(coeffs, [np.pi, -np.pi, -np.pi, np.pi], atol=1e-10)

    # All coefficients should be nonzero after normalization
    normalized = coeffs / len(phases)
    assert all(abs(a) > 1e-10 for a in normalized), (
        "All Walsh terms should be nonzero for this CZ-like diagonal"
    )


# ---------------------------------------------------------------------------
# Circuit correctness tests
# ---------------------------------------------------------------------------


def _walsh_circuit_test(diagonal: np.ndarray) -> None:
    """Verify that the Walsh circuit implements the target diagonal unitary.

    Simulates the synthesized circuit and compares the resulting unitary
    matrix against np.diag(diagonal) up to global phase. Uses little-endian
    convention where qs[0] is the least significant bit.
    """
    n = int(np.log2(len(diagonal)))
    circ = diagonal_unitary_walsh(diagonal)
    actual = get_unitary(circ, n, endianness=Endianness.LITTLE)
    assert_allclose_ignorephase(np.diag(diagonal), actual)


@pytest.mark.parametrize(
    "diagonal",
    [
        np.array([1.0, -1.0]),  # Z gate
        np.array([1.0, 1j]),  # S-like
        np.exp(1j * np.array([0.3, -0.7])),  # arbitrary phases
    ],
)
def test_diagonal_unitary_walsh_1q(diagonal: np.ndarray) -> None:
    """Test Walsh synthesis for 1-qubit unitary diagonals."""
    _walsh_circuit_test(diagonal)


@pytest.mark.parametrize(
    "diagonal",
    [
        np.array([1.0, 1.0, 1.0, -1.0]),  # CZ
        np.exp(1j * np.array([0.1, 0.5, -0.3, 0.8])),  # generic
    ],
)
def test_diagonal_unitary_walsh_2q(diagonal: np.ndarray) -> None:
    """Test Walsh synthesis for 2-qubit unitary diagonals."""
    _walsh_circuit_test(diagonal)


@pytest.mark.parametrize(
    "diagonal",
    [
        np.exp(1j * np.array([0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, -0.8])),
    ],
)
def test_diagonal_unitary_walsh_3q(diagonal: np.ndarray) -> None:
    """Test Walsh synthesis for 3-qubit unitary diagonals."""
    _walsh_circuit_test(diagonal)


def test_diagonal_unitary_walsh_invalid_non_unitary() -> None:
    """diagonal_unitary_walsh should raise ValueError for non-unitary input."""
    with pytest.raises(ValueError, match="magnitude 1"):
        diagonal_unitary_walsh(np.array([1.0, 2.0]))


def test_diagonal_unitary_walsh_invalid_length() -> None:
    """diagonal_unitary_walsh should raise ValueError for non-power-of-2 length."""
    with pytest.raises(ValueError, match="power of 2"):
        diagonal_unitary_walsh(np.array([1.0, -1.0, 1.0]))


def test_diagonal_unitary_walsh_invalid_empty() -> None:
    """diagonal_unitary_walsh should reject an empty diagonal cleanly."""
    with pytest.raises(ValueError, match="power of 2"):
        diagonal_unitary_walsh(np.array([]))
