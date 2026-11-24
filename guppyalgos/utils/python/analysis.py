"""Analysis helpers for phase-estimation notebooks and tests.

The Hamiltonian-analysis routine builds dense matrices and diagonalizes them,
so it scales exponentially in the number of qubits.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import zixy.qubit.pauli as zqp
from numpy.typing import NDArray
from scipy.linalg import expm

from .binary import fixed_point_to_float
from .trotter import (
    dominant_trotterized_eigenphase,
    trotter_step_matrix,
    trotterized_eigenphases,
)


def analyze_qubit_pauli_operator(
    ham_op: zqp.RealTermSum,
    time_step: float,
    prepared_state: NDArray[np.complex128] | None = None,
    little_endian: bool = True,
) -> pd.DataFrame:
    """Summarize exact and Trotterized spectra for a Pauli Hamiltonian.

    The returned single-row dataframe keeps the dense matrices, spectra, and
    overlap diagnostics together so the notebook can display them compactly.

    This routine constructs dense ``2**n`` matrices and diagonalizes them, so
    the cost is exponential in the number of qubits.
    """
    ham_mat = ham_op.to_sparse_matrix(little_endian).toarray()
    energies, eigenvectors = np.linalg.eigh(ham_mat)
    exact_step = expm(-1j * (0.5 * np.pi * time_step) * ham_mat)
    trotter_step = trotter_step_matrix(ham_op, time_step, little_endian)
    trotter_phases, trotter_eigenvectors = trotterized_eigenphases(
        ham_op, time_step, little_endian
    )

    row: dict[str, object] = {
        "ham_mat": ham_mat,
        "energies": energies,
        "eigenvectors": eigenvectors,
        "exact_step": exact_step,
        "trotter_step": trotter_step,
        "trotter_step_error": float(np.linalg.norm(trotter_step - exact_step)),
        "trotter_phases": trotter_phases,
        "trotter_eigenvectors": trotter_eigenvectors,
    }

    if prepared_state is not None:
        prepared_state = np.asarray(prepared_state, dtype=np.complex128)
        exact_overlaps = np.abs(eigenvectors.conj().T @ prepared_state) ** 2
        target_index = int(np.argmax(exact_overlaps))
        target_energy = float(energies[target_index])
        target_phase = float((-target_energy * time_step / 2) % 2)
        trotter_overlaps = np.abs(trotter_eigenvectors.conj().T @ prepared_state) ** 2
        dominant_trotter_phase, dominant_trotter_overlap = (
            dominant_trotterized_eigenphase(
                ham_op, time_step, prepared_state, little_endian
            )
        )
        row.update(
            {
                "prepared_state": prepared_state,
                "exact_overlaps": exact_overlaps,
                "target_index": target_index,
                "target_energy": target_energy,
                "target_phase": target_phase,
                "trotter_overlaps": trotter_overlaps,
                "dominant_trotter_phase": dominant_trotter_phase,
                "dominant_trotter_overlap": dominant_trotter_overlap,
            }
        )

    return pd.DataFrame.from_records([row])


def dominant_trotter_phase(analysis: pd.DataFrame) -> tuple[float, float]:
    """Return the dominant Trotter phase and overlap from an analysis table."""
    row = analysis.iloc[0]
    return float(row["dominant_trotter_phase"]), float(row["dominant_trotter_overlap"])


def measurement_dataframe(
    result_counter: Mapping[str, int], int_bits: int = 1
) -> pd.DataFrame:
    """Store a measurement histogram and decoded phases as a dataframe.

    The measurement bitstrings are interpreted as little-endian by default,
    which matches the current QPE readout convention.
    """
    total_counts = sum(result_counter.values())
    rows = [
        {
            "bitstring": bitstring,
            "counts": count,
            "phase": fixed_point_to_float([bit == "1" for bit in bitstring], int_bits),
            "empirical_probability": count / total_counts,
        }
        for bitstring, count in sorted(
            result_counter.items(), key=lambda item: item[1], reverse=True
        )
    ]
    return pd.DataFrame.from_records(rows)


def dominant_measured_phase(
    result_counter: Mapping[str, int],
    int_bits: int = 1,
) -> tuple[str, int, float]:
    """Return the dominant bitstring and its decoded phase from a histogram."""
    counts_df = measurement_dataframe(result_counter, int_bits=int_bits)
    dominant_row = counts_df.iloc[0]
    return (
        str(dominant_row["bitstring"]),
        int(dominant_row["counts"]),
        float(dominant_row["phase"]),
    )


__all__ = [
    "analyze_qubit_pauli_operator",
    "dominant_measured_phase",
    "dominant_trotter_phase",
    "measurement_dataframe",
]
