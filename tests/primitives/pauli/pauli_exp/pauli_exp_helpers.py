"""Pauli-exponential helper functions for tests."""

import numpy as np
import zixy.qubit.pauli as zqp
from numpy.typing import NDArray
from scipy.linalg import expm


def pauli_exp_matrix(
    pauli_string: zqp.String,
    n_state_qubits: int,
    theta: float,
    little_endian: bool = True,
) -> NDArray[np.complex128]:
    """Return the classical reference matrix for ``exp(-i pi theta P / 2)``."""
    pauli_mat = (
        zqp.RealTermSum.from_str(str(pauli_string), n_state_qubits)
        .to_sparse_matrix(little_endian)
        .toarray()
    )
    return expm(-1j * (0.5 * np.pi * theta) * pauli_mat)
