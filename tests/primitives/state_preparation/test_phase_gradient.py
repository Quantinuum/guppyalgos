"""Tests for the phase gradient (Fourier) state preparation."""

from __future__ import annotations

from typing import no_type_check

import numpy as np
import pytest
from guppylang import guppy
from guppylang.std.debug import state_result
from guppylang.std.quantum import discard_array

from guppyalgos.primitives.state_preparation.phase_gradient import (
    Convention,
    phase_gradient,
)
from guppyalgos.utils import qarray
from guppyalgos.testing import (
    assert_allclose_ignorephase,
    get_statevector,
    switch_endianness,
)


def phase_gradient_state_expected_standard(n_qubits: int) -> np.ndarray:
    """Return the expected standard-convention phase-gradient statevector.

    This is the little-endian arithmetic/Gidney layout used by the
    phase-gradient rotation algorithm. The positive phase schedule is reversed
    so qubit 0 is the least-significant qubit for addition.
    """
    N = 2**n_qubits
    state = np.ones(N, dtype=complex) / np.sqrt(N)
    for j in range(N):
        phase = 0.0
        for bit in range(n_qubits):
            if (j >> (n_qubits - 1 - bit)) & 1:
                phase += np.pi / (2 ** (n_qubits - 1 - bit))
        state[j] *= np.exp(1j * phase)
    return state


def phase_gradient_state_expected_reciprocal(n_qubits: int) -> np.ndarray:
    """Return the expected reciprocal-convention phase-gradient statevector.

    This is the big-endian direct-index layout.

    The state is:

        |F⟩ = (1/√2^n) ⊗_{k=0}^{n-1} (|0⟩ + e^{iπ/2^k}|1⟩)

    which in the computational basis has amplitude e^{iπ·k/2^(n-1)} / √(2^n)
    for basis state |k⟩ in standard basis ordering.

    Args:
        n_qubits: Number of qubits.

    Returns:
        Complex numpy array of length 2**n_qubits.

    """
    N = 2**n_qubits
    state = np.ones(N, dtype=complex) / np.sqrt(N)
    for j in range(N):
        phase = 0.0
        for bit in range(n_qubits):
            if (j >> (n_qubits - 1 - bit)) & 1:
                phase += np.pi / (2**bit)
        state[j] *= np.exp(1j * phase)
    return state


@pytest.mark.parametrize(
    ("convention", "expected_fn"),
    [
        (Convention.Standard, phase_gradient_state_expected_standard),
        (Convention.Reciprocal, phase_gradient_state_expected_reciprocal),
    ],
)
@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5, 6])
@no_type_check
def test_phase_gradient_statevector(
    n_qubits: int,
    convention: Convention,
    expected_fn: callable,
) -> None:
    """Test that phase_gradient produces the statevector for each convention."""
    prep = phase_gradient(n_qubits, convention=convention)

    @guppy
    def main() -> None:
        qs = qarray(n_qubits)
        prep(qs)
        state_result("result_state", qs)
        discard_array(qs)

    simulated = switch_endianness(get_statevector(main, n_qubits))
    expected = expected_fn(n_qubits)
    assert_allclose_ignorephase(simulated, expected)
