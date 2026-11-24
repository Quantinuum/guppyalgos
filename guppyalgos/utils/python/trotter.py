"""Classical helpers for Trotterized Hamiltonian analysis.

These routines build and diagonalize dense matrices, so they scale
exponentially in the number of qubits and are intended for small systems,
tests, and notebook analysis.
"""

from __future__ import annotations

import numpy as np
import zixy.qubit.pauli as zqp
from numpy.typing import NDArray
from scipy.linalg import expm


def trotter_step_matrix(
    ham_op: zqp.RealTermSum,
    time_step: float,
    little_endian: bool = True,
) -> NDArray[np.complex128]:
    """Return the one-step first-order Trotter unitary matrix.

    Args:
        ham_op: Pauli Hamiltonian to exponentiate term-by-term.
        time_step: Dimensionless Trotter time step.
        little_endian: Whether the underlying matrix convention is little-endian.

    Returns:
        The dense unitary matrix for one first-order Trotter step.

    """
    n_qubits = len(ham_op.qubits)
    trotter_step = np.eye(2**n_qubits, dtype=np.complex128)

    ham_terms: list[zqp.RealTerm] = list(ham_op.to_terms())  # ty: ignore[invalid-assignment]

    for term in ham_terms:
        if term.string.is_identity():
            trotter_step = (
                np.exp(-1j * (0.5 * np.pi * time_step) * term.coeff) * trotter_step
            )
        else:
            term_matrix = term.to_sparse_matrix(little_endian).toarray()
            trotter_step = (
                expm(-1j * (0.5 * np.pi * time_step) * term_matrix) @ trotter_step
            )

    return trotter_step


def trotterized_eigenphases(
    ham_op: zqp.RealTermSum,
    time_step: float,
    little_endian: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Return sorted one-step Trotter eigenphases and eigenvectors.

    This diagonalizes the dense Trotter step matrix, so the cost is exponential
    in the number of qubits.

    Args:
        ham_op: Pauli Hamiltonian to diagonalize through the Trotter step.
        time_step: Dimensionless Trotter time step.
        little_endian: Whether the underlying matrix convention is little-endian.

    Returns:
        A tuple of the sorted eigenphases in ``[0, 2)`` and the corresponding
        eigenvectors as columns.

    """
    trotter_u = trotter_step_matrix(ham_op, time_step, little_endian)
    eigvals, eigvecs = np.linalg.eig(trotter_u)
    phases = (np.angle(eigvals) / np.pi) % 2
    order = np.argsort(phases)
    return phases[order], eigvecs[:, order]


def dominant_trotterized_eigenphase(
    ham_op: zqp.RealTermSum,
    time_step: float,
    prepared_state: NDArray[np.complex128],
    little_endian: bool = True,
) -> tuple[float, float]:
    """Return the Trotter eigenphase with the largest overlap to a state.

    This performs trotterized_eigenphases, so it is exponentially
    expensive in the number of qubits.

    Args:
        ham_op: Pauli Hamiltonian whose Trotter step is analyzed.
        time_step: Dimensionless Trotter time step.
        prepared_state: State vector to project onto the Trotter eigenbasis.
        little_endian: Whether the underlying matrix convention is little-endian.

    Returns:
        The dominant eigenphase and the corresponding overlap probability.

    """
    phases, eigvecs = trotterized_eigenphases(ham_op, time_step, little_endian)
    prepared_state = np.asarray(prepared_state, dtype=np.complex128)
    overlaps = np.abs(eigvecs.conj().T @ prepared_state) ** 2
    idx = int(np.argmax(overlaps))
    return float(phases[idx]), float(overlaps[idx])
