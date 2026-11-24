r"""Pauli-observable statistics and measurement helpers.

For :math:`H = \sum_j c_j P_j`, estimate real observables from binary samples
and build Guppy programs for indirect or direct system-register measurements.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from typing import TextIO, cast, no_type_check

import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, comptime, nat, output, owned
from guppylang.std.quantum import collect_measurements, measure, measure_array, qubit

from guppyalgos.algorithms.phase_estimation import hadamard_test
from guppyalgos.primitives.pauli import pauli_to_cntrl_gate, pauli_to_z_basis

from .stats import (
    BinaryShotEstimate,
    estimate_expectation_from_binary_samples,
)


@dataclass(frozen=True)
class PauliObservableExpectationEstimate:
    r"""Statistics for a real Pauli observable.

    For :math:`H = \sum_j c_jP_j`, stores :math:`\hat E[H]`, propagated
    variance :math:`\widehat{\operatorname{Var}}(H)`, and per-term estimates.
    """

    expectation: float
    variance: float
    standard_error: float
    term_estimates: dict[str, BinaryShotEstimate]

    def _pretty_term_estimates(self) -> str:
        """Return a formatted table of per-term estimates."""
        term_width = max(
            6,
            max((len(term or "I") for term in self.term_estimates), default=1),
        )
        header = (
            f"{'term':<{term_width}}  "
            f"{'exp':>10}  "
            f"{'var':>10}  "
            f"{'stderr':>10}  "
            f"{'shots':>5}  "
            f"{'+':>4}  "
            f"{'-':>4}"
        )
        rows = [
            estimate._pretty(term or "I", term_width=term_width)
            for term, estimate in sorted(
                self.term_estimates.items(), key=lambda kv: (kv[0] != "", kv[0])
            )
        ]
        return "\n".join([header, *rows])

    def print_terms(self, file: TextIO | None = None) -> None:
        """Print a formatted table of per-term estimates."""
        print(self._pretty_term_estimates(), file=file)


def _ensure_real_pauli_observable(operator: object) -> zqp.RealTermSum:
    """Validate that ``operator`` is a real Pauli observable."""
    if not isinstance(operator, zqp.RealTermSum):
        raise TypeError(
            "Pauli observable estimation only supports zixy.RealTermSum "
            "objects with real coefficients. Use real_part/imag_part for "
            "complex operators."
        )
    return operator


def estimate_pauli_observable_expectation_from_binary_samples(
    operator: zqp.RealTermSum,
    pauli_samples: Mapping[str, Sequence[bool] | Mapping[bool, int]],
) -> PauliObservableExpectationEstimate:
    """Estimate statistics for a real (hermitian) Pauli observable from binary samples.

    The operator must be a :class:`zixy.qubit.pauli.RealTermSum`, i.e. a sum of
    Pauli strings with real coefficients. Each term label in ``pauli_samples``
    should match the string form of the Pauli term being sampled. Zero-
    coefficient terms are treated as no-ops and do not require sample data.
    Additional keys in ``pauli_samples`` are allowed and ignored, which lets
    callers reuse a broader sample collection than the estimator needs.

    For ``H = sum_j c_j P_j``, this returns the estimated mean

        E_hat(H) = sum_j c_j * mu_hat_j

    and the propagated variance

        Var_hat(H) = sum_j c_j**2 * var_hat_j.
    """
    operator = _ensure_real_pauli_observable(operator)
    term_estimates: dict[str, BinaryShotEstimate] = {}
    expectation = 0.0
    variance = 0.0

    for term in operator.to_terms():
        pauli_string = term.cmpnt
        pauli_key = str(pauli_string)
        coeff = float(term.coeff)

        if not pauli_key:  # identity term
            term_estimate = BinaryShotEstimate(
                expectation=1.0,
                variance=0.0,
                standard_error=0.0,
                shots=0,
                positive_shots=0,
                negative_shots=0,
            )
        elif coeff == 0.0:
            term_estimate = BinaryShotEstimate(
                expectation=0.0,
                variance=0.0,
                standard_error=0.0,
                shots=0,
                positive_shots=0,
                negative_shots=0,
            )
        else:
            try:
                term_samples = pauli_samples[pauli_key]
            except KeyError as exc:
                raise KeyError(
                    f"Missing samples for Pauli string {pauli_key!r}"
                ) from exc
            term_estimate = estimate_expectation_from_binary_samples(term_samples)

        term_estimates[pauli_key] = term_estimate
        expectation += coeff * term_estimate.expectation
        variance += (coeff * coeff) * term_estimate.variance

    return PauliObservableExpectationEstimate(
        expectation=expectation,
        variance=variance,
        standard_error=sqrt(variance),
        term_estimates=term_estimates,
    )


def _parity_from_bitstring(bitstring: Sequence[bool], indices: Sequence[int]) -> bool:
    """Return the XOR parity of selected positions in a measured bitstring."""
    parity = False
    for index in indices:
        try:
            parity ^= bitstring[index]
        except IndexError as exc:
            raise ValueError(
                f"Bitstring of length {len(bitstring)} does not contain selected "
                f"index {index}"
            ) from exc
    return parity


def _bitstring_samples_to_parities(
    indices: Sequence[int],
    bitstring_samples: Sequence[Sequence[bool]] | Mapping[tuple[bool, ...], int],
) -> Sequence[bool] | Mapping[bool, int]:
    """Convert full-register bitstring samples into selected-index parities."""
    if isinstance(bitstring_samples, Mapping):
        bitstring_counts = cast(Mapping[tuple[bool, ...], int], bitstring_samples)
        parity_counts: Counter[bool] = Counter()
        for bitstring, count in bitstring_counts.items():
            if count < 0:
                raise ValueError(
                    "Bitstring sample histogram counts must be non-negative"
                )
            parity_counts[_parity_from_bitstring(bitstring, indices)] += count
        return parity_counts

    return [
        _parity_from_bitstring(bitstring, indices) for bitstring in bitstring_samples
    ]


def estimate_pauli_observable_expectation_from_bitstrings(
    operator: zqp.RealTermSum,
    pauli_bitstrings: Mapping[
        str,
        Sequence[Sequence[bool]] | Mapping[tuple[bool, ...], int],
    ],
) -> PauliObservableExpectationEstimate:
    r"""Estimate a real Pauli observable from full-register bitstrings.

    ``pauli_bitstrings`` maps each non-identity term to either a sequence of
    full-register bitstrings, one per shot, or a histogram mapping bitstring
    tuples to non-negative counts.
    For :math:`P_j`, each shot becomes
    :math:`(-1)^{\oplus_{k\in\operatorname{supp}(P_j)} b_k}` before applying
    the binary estimator. Variance propagation assumes independent term samples;
    grouped settings require covariance terms.
    """
    operator = _ensure_real_pauli_observable(operator)
    parity_samples: dict[str, Sequence[bool] | Mapping[bool, int]] = {}

    for term in operator.to_terms():
        pauli_key = str(term.cmpnt)
        if not pauli_key or float(term.coeff) == 0.0:
            continue

        try:
            bitstrings = pauli_bitstrings[pauli_key]
        except KeyError as exc:
            raise KeyError(
                f"Missing bitstrings for Pauli string {pauli_key!r}"
            ) from exc

        parity_samples[pauli_key] = _bitstring_samples_to_parities(
            tuple(cast(zqp.RealTerm, term).string.get_dict()), bitstrings
        )

    return estimate_pauli_observable_expectation_from_binary_samples(
        operator, parity_samples
    )


def make_hadamard_test_pauli[n_q: nat](
    pauli_string: zqp.String,
    size: int,
) -> GuppyFunctionDefinition[[array[qubit, n_q]], None]:
    """Return a Hadamard-test program specialized to a Pauli string.

    This Python factory returns a Guppy-compiled program that allocates the
    ancilla internally, applies the Hadamard-test primitive for operator
    kickback, and records the ancilla outcome in the ``ancilla`` result
    stream. Repeated shots estimate the expectation value of the observable
    associated with ``pauli_string``.
    """
    controlled_pauli = pauli_to_cntrl_gate(pauli_string, size)

    @guppy
    @no_type_check
    def hadamard_test_pauli(
        unitary_regs: array[qubit, comptime(size)],
    ) -> None:
        ancilla = qubit()
        hadamard_test(ancilla, unitary_regs, controlled_pauli)
        output("ancilla", measure(ancilla).read())

    return hadamard_test_pauli


def make_direct_measure_pauli[n_q: nat](
    measurement_structure_function: GuppyFunctionDefinition[[array[qubit, n_q]], None],
    size: int,
) -> GuppyFunctionDefinition[[array[qubit, n_q] @ owned], None]:  # ty: ignore[not-subscriptable]
    """Build a full-register measurement program.

    ``measurement_structure_function`` prepares the register for Z-basis
    measurement. The returned program emits the bitstring on ``"bitstring"``
    and consumes the register.
    """

    @guppy
    @no_type_check
    def direct_measure(
        qreg: array[qubit, comptime(size)] @ owned,
    ) -> None:
        measurement_structure_function(qreg)
        output("bitstring", collect_measurements(measure_array(qreg)))

    return direct_measure


def make_direct_measure_pauli_simple[n_q: nat](
    pauli_string: zqp.String,
    size: int,
) -> GuppyFunctionDefinition[[array[qubit, n_q] @ owned], None]:  # ty: ignore[not-subscriptable]
    """Build a direct measurement program with per-term Z-basis rotations."""
    basis_change = pauli_to_z_basis(pauli_string, size)
    return make_direct_measure_pauli(basis_change, size)


__all__ = [
    "PauliObservableExpectationEstimate",
    "estimate_pauli_observable_expectation_from_binary_samples",
    "estimate_pauli_observable_expectation_from_bitstrings",
    "make_direct_measure_pauli",
    "make_direct_measure_pauli_simple",
    "make_hadamard_test_pauli",
]
