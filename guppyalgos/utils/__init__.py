"""Classical and Guppy utility functions."""

from .guppy.array import discard_nested_arrays
from .guppy.math import floor, get_bit, tan
from .guppy.unsafe_borrow import _unsafe_array_borrow, _unsafe_array_unborrow
from .guppy.gates import (
    apply_bitstring,
    apply_phase,
    ccswap,
    cphase,
    cswap,
    qarray,
    t_state,
    transversal,
)
from .python.analysis import (
    analyze_qubit_pauli_operator,
    dominant_measured_phase,
    dominant_trotter_phase,
    measurement_dataframe,
)
from .python.angle_finder import (
    BaseCompilePhases,
    ChebyshevPolynomial,
    CompilerPhasesNumba,
    CompilerPhasesNumpy,
    FourierPolynomial,
    FunctionParity,
    GQSPAngleFinder,
    QSPAngleFinder,
    qsp_phase_reflection,
)
from .python.trotter import (
    dominant_trotterized_eigenphase,
    trotter_step_matrix,
    trotterized_eigenphases,
)
from .python.binary import (
    bits_to_int,
    fixed_point_to_float,
    float_to_fixed_point,
    int_to_bits,
)
from .python.phase import (
    binary_fraction,
    phase_distance_mod_2,
    phase_to_energy_qpe,
    phase_to_energy_qubitized_qpe,
)

__all__ = [
    "BaseCompilePhases",
    "ChebyshevPolynomial",
    "CompilerPhasesNumba",
    "CompilerPhasesNumpy",
    "FourierPolynomial",
    "FunctionParity",
    "GQSPAngleFinder",
    "QSPAngleFinder",
    "_unsafe_array_borrow",
    "_unsafe_array_unborrow",
    "analyze_qubit_pauli_operator",
    "apply_bitstring",
    "apply_phase",
    "binary_fraction",
    "bits_to_int",
    "ccswap",
    "cphase",
    "cswap",
    "discard_nested_arrays",
    "dominant_measured_phase",
    "dominant_trotter_phase",
    "dominant_trotterized_eigenphase",
    "fixed_point_to_float",
    "float_to_fixed_point",
    "floor",
    "get_bit",
    "int_to_bits",
    "measurement_dataframe",
    "phase_distance_mod_2",
    "phase_to_energy_qpe",
    "phase_to_energy_qubitized_qpe",
    "qarray",
    "qsp_phase_reflection",
    "t_state",
    "tan",
    "transversal",
    "trotter_step_matrix",
    "trotterized_eigenphases",
]
