"""Tests for operator averaging helpers."""

from __future__ import annotations

from collections import Counter
from io import StringIO
from math import sqrt
from typing import no_type_check

import pytest
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.std.builtins import array, comptime
from guppylang.std.quantum import discard_array, h, qubit, s, x

from guppyalgos.primitives.measurement.pauli import (
    PauliObservableExpectationEstimate,
    estimate_pauli_observable_expectation_from_binary_samples,
    estimate_pauli_observable_expectation_from_bitstrings,
    make_direct_measure_pauli,
    make_direct_measure_pauli_simple,
    make_hadamard_test_pauli,
)
from guppyalgos.primitives.measurement.stats import (
    BinaryShotEstimate,
    estimate_expectation_from_binary_samples,
)
from guppyalgos.utils import qarray


def test_estimate_expectation_from_binary_samples_sequence() -> None:
    """Sequence-based sample data should map to the +/- 1 observable."""
    estimate = estimate_expectation_from_binary_samples([False, False, True])

    assert estimate.shots == 3
    assert estimate.positive_shots == 2
    assert estimate.negative_shots == 1
    assert estimate.expectation == pytest.approx(1 / 3)
    assert estimate.variance == pytest.approx(8 / 27)
    assert estimate.standard_error == pytest.approx(sqrt(8 / 27))


def test_estimate_expectation_from_binary_samples_negative() -> None:
    """Binary samples with more ``True`` outcomes should yield a negative mean."""
    estimate = estimate_expectation_from_binary_samples([True, True, False])

    assert estimate.expectation == pytest.approx(-1 / 3)
    assert estimate.variance == pytest.approx(8 / 27)


def test_estimate_expectation_from_binary_samples_histogram() -> None:
    """Histogram-based sample data should be accepted when counts are valid."""
    estimate = estimate_expectation_from_binary_samples({False: 2, True: 1})

    assert estimate.shots == 3
    assert estimate.positive_shots == 2
    assert estimate.negative_shots == 1
    assert estimate.expectation == pytest.approx(1 / 3)
    assert estimate.variance == pytest.approx(8 / 27)


def test_binary_histogram_counts_must_be_non_negative() -> None:
    """Histogram counts must not be negative."""
    with pytest.raises(ValueError, match="non-negative"):
        estimate_expectation_from_binary_samples({False: -1})


def test_binary_shot_estimate_pretty() -> None:
    """A binary shot estimate should format as a compact table row."""
    estimate = BinaryShotEstimate(
        expectation=0.25,
        variance=0.75,
        standard_error=sqrt(0.75),
        shots=8,
        positive_shots=5,
        negative_shots=3,
    )

    assert estimate._pretty("Z0") == (
        "Z0       +0.250000    0.750000    0.866025      8     5     3"
    )


def test_pauli_observable_expectation_print_terms_formats_sorted_table() -> None:
    """The term table should keep identity first and render aligned rows."""
    estimate = PauliObservableExpectationEstimate(
        expectation=0.0,
        variance=0.0,
        standard_error=0.0,
        term_estimates={
            "": BinaryShotEstimate(1.0, 0.0, 0.0, 0, 0, 0),
            "X0 X1 Y2 Y3": BinaryShotEstimate(0.5, 0.25, 0.5, 4, 3, 1),
            "Z0": BinaryShotEstimate(-0.25, 0.9375, 0.9682458365518543, 8, 3, 5),
        },
    )
    term_width = max(6, max(len(term or "I") for term in estimate.term_estimates))

    buf = StringIO()
    estimate.print_terms(file=buf)

    lines = buf.getvalue().splitlines()

    assert lines[0].split() == ["term", "exp", "var", "stderr", "shots", "+", "-"]
    assert lines[1:] == [
        estimate.term_estimates[""]._pretty("I", term_width=term_width),
        estimate.term_estimates["X0 X1 Y2 Y3"]._pretty(
            "X0 X1 Y2 Y3", term_width=term_width
        ),
        estimate.term_estimates["Z0"]._pretty("Z0", term_width=term_width),
    ]


def test_estimate_pauli_observable_expectation_from_binary_samples() -> None:
    """A weighted real term sum should propagate term expectations and variance."""
    hamiltonian = zqp.RealTermSum.from_str("(0.5, Z0), (-0.25, X1), (0.125, I0)", 2)
    estimate = estimate_pauli_observable_expectation_from_binary_samples(
        hamiltonian,
        {
            "Z0": [False, True],
            "X1": Counter({False: 2}),
        },
    )

    assert set(estimate.term_estimates) == {"", "X1", "Z0"}
    assert estimate.term_estimates[""].expectation == pytest.approx(1.0)
    assert estimate.term_estimates[""].variance == pytest.approx(0.0)
    assert estimate.term_estimates["Z0"].expectation == pytest.approx(0.0)
    assert estimate.term_estimates["Z0"].variance == pytest.approx(0.5)
    assert estimate.term_estimates["X1"].expectation == pytest.approx(1.0)
    assert estimate.term_estimates["X1"].variance == pytest.approx(0.0)
    assert estimate.expectation == pytest.approx(-0.125)
    assert estimate.variance == pytest.approx(0.125)
    assert estimate.standard_error == pytest.approx(sqrt(0.125))


def test_binary_observable_estimate_handles_four_qubit_multi_body_terms() -> None:
    """Use sparse supports for multi-body terms on a four-qubit register."""
    hamiltonian = zqp.RealTermSum.from_str(
        "(0.5, Z0 Z2), (-0.25, X1 Y3), (0.125, X0 Y1 Z2), (0.2, I0)",
        4,
    )
    estimate = estimate_pauli_observable_expectation_from_binary_samples(
        hamiltonian,
        {
            "Z0 Z2": [False, False, True],
            "X1 Y3": [False, True, True],
            "X0 Y1 Z2": [False, True, True],
        },
    )

    assert estimate.term_estimates["Z0 Z2"].expectation == pytest.approx(1 / 3)
    assert estimate.term_estimates["X1 Y3"].expectation == pytest.approx(-1 / 3)
    assert estimate.term_estimates["X0 Y1 Z2"].expectation == pytest.approx(-1 / 3)
    assert estimate.expectation == pytest.approx(49 / 120)
    assert estimate.variance == pytest.approx(7 / 72)
    assert estimate.standard_error == pytest.approx(sqrt(7 / 72))


def test_estimate_pauli_observable_expectation_from_binary_samples_one_term() -> None:
    """A single Pauli term should reduce to the corresponding binary estimate."""
    hamiltonian = zqp.RealTermSum.from_str("(0.5, Z0)", 1)
    estimate = estimate_pauli_observable_expectation_from_binary_samples(
        hamiltonian,
        {"Z0": [False, False, True]},
    )

    assert set(estimate.term_estimates) == {"Z0"}
    assert estimate.term_estimates["Z0"].expectation == pytest.approx(1 / 3)
    assert estimate.term_estimates["Z0"].variance == pytest.approx(8 / 27)
    assert estimate.expectation == pytest.approx(1 / 6)
    assert estimate.variance == pytest.approx(2 / 27)
    assert estimate.standard_error == pytest.approx(sqrt(2 / 27))


def test_estimate_pauli_observable_expectation_from_bitstrings() -> None:
    """Extract Pauli parities from full-register samples."""
    hamiltonian = zqp.RealTermSum.from_str("(0.5, Z0), (-0.25, X1), (0.125, I0)", 2)
    estimate = estimate_pauli_observable_expectation_from_bitstrings(
        hamiltonian,
        {
            "Z0": [[False, False], [True, False]],
            "X1": [[False, False], [False, False]],
        },
    )

    assert estimate.term_estimates["Z0"].expectation == pytest.approx(0.0)
    assert estimate.term_estimates["X1"].expectation == pytest.approx(1.0)
    assert estimate.expectation == pytest.approx(-0.125)
    assert estimate.variance == pytest.approx(0.125)


def test_bitstring_parities_handle_four_qubit_multi_body_terms() -> None:
    """Extract non-adjacent multi-body parities from full-register samples."""
    hamiltonian = zqp.RealTermSum.from_str(
        "(1.0, X0 Y2 Z3), (-0.5, Y0 X1 Z2)",
        4,
    )
    estimate = estimate_pauli_observable_expectation_from_bitstrings(
        hamiltonian,
        {
            "X0 Y2 Z3": [
                [False, False, False, False],
                [True, False, True, True],
                [True, False, True, False],
            ],
            "Y0 X1 Z2": [
                [False, False, False, False],
                [True, True, False, False],
                [False, True, True, False],
            ],
        },
    )

    assert estimate.term_estimates["X0 Y2 Z3"].expectation == pytest.approx(1 / 3)
    assert estimate.term_estimates["Y0 X1 Z2"].expectation == pytest.approx(1.0)
    assert estimate.expectation == pytest.approx(-1 / 6)
    assert estimate.variance == pytest.approx(8 / 27)


def test_estimate_pauli_observable_expectation_from_bitstrings_histogram() -> None:
    """Reduce full-register histograms to parity histograms."""
    hamiltonian = zqp.RealTermSum.from_str("(1.0, Z0 Z1)", 2)
    estimate = estimate_pauli_observable_expectation_from_bitstrings(
        hamiltonian,
        {"Z0 Z1": Counter({(False, False): 2, (True, False): 1})},
    )

    assert estimate.term_estimates["Z0 Z1"].expectation == pytest.approx(1 / 3)
    assert estimate.term_estimates["Z0 Z1"].variance == pytest.approx(8 / 27)


def test_bitstring_parity_histogram_handles_four_qubit_support() -> None:
    """Reduce histograms for sparse, non-adjacent support."""
    hamiltonian = zqp.RealTermSum.from_str("(1.0, Z0 Y2 X3)", 4)
    estimate = estimate_pauli_observable_expectation_from_bitstrings(
        hamiltonian,
        {
            "Z0 Y2 X3": Counter(
                {
                    (False, False, False, False): 2,
                    (True, False, True, False): 1,
                    (True, False, False, False): 1,
                }
            )
        },
    )

    assert estimate.term_estimates["Z0 Y2 X3"].expectation == pytest.approx(0.5)
    assert estimate.term_estimates["Z0 Y2 X3"].variance == pytest.approx(3 / 16)


def test_bitstring_parity_rejects_short_bitstring() -> None:
    """Reject bitstrings missing a qubit in the Pauli support."""
    hamiltonian = zqp.RealTermSum.from_str("(1.0, Z0 Z1)", 2)

    with pytest.raises(ValueError, match="does not contain selected index 1"):
        estimate_pauli_observable_expectation_from_bitstrings(
            hamiltonian,
            {"Z0 Z1": [[False]]},
        )


def test_bitstring_parity_rejects_short_bitstring_for_four_qubit_support() -> None:
    """Reject short bitstrings for high-index multi-body terms."""
    hamiltonian = zqp.RealTermSum.from_str("(1.0, X0 Y2 Z3)", 4)

    with pytest.raises(ValueError, match="does not contain selected index 3"):
        estimate_pauli_observable_expectation_from_bitstrings(
            hamiltonian,
            {"X0 Y2 Z3": [[False, False, False]]},
        )


def test_estimate_pauli_observable_rejects_complex_operator() -> None:
    """Complex Pauli operators should be rejected explicitly."""
    complex_operator = zqp.ComplexTermSum.from_str("(1j, Z0)", 1)

    with pytest.raises(TypeError, match="real coefficients"):
        estimate_pauli_observable_expectation_from_binary_samples(
            complex_operator,
            {"Z0": [False]},
        )


def test_estimate_pauli_observable_expectation_skips_zero_coeff_terms() -> None:
    """Zero-coefficient terms should not require sample data."""
    hamiltonian = zqp.RealTermSum.from_str("(0.5, Z0), (0.0, X1), (0.125, I0)", 2)
    estimate = estimate_pauli_observable_expectation_from_binary_samples(
        hamiltonian,
        {
            "Z0": [False, True],
        },
    )

    assert set(estimate.term_estimates) == {"", "X1", "Z0"}
    assert estimate.term_estimates["X1"].expectation == pytest.approx(0.0)
    assert estimate.term_estimates["X1"].variance == pytest.approx(0.0)
    assert estimate.term_estimates["X1"].shots == 0
    assert estimate.expectation == pytest.approx(0.125)
    assert estimate.variance == pytest.approx(0.125)


def test_binary_observable_estimate_skips_zero_coeff_four_qubit_term() -> None:
    """Zero-coefficient multi-body terms need no samples."""
    hamiltonian = zqp.RealTermSum.from_str(
        "(0.5, Z0 Z2), (0.0, X1 Y3), (0.125, I0)",
        4,
    )
    estimate = estimate_pauli_observable_expectation_from_binary_samples(
        hamiltonian,
        {"Z0 Z2": [False, True]},
    )

    assert estimate.term_estimates["X1 Y3"].shots == 0
    assert estimate.expectation == pytest.approx(0.125)


def test_make_hadamard_test_pauli_identity_measures_zero() -> None:
    """The identity Pauli should leave the ancilla in ``|0>``."""
    pauli_string = zqp.String.from_str("I0", 1)
    hadamard_test_pauli = make_hadamard_test_pauli(pauli_string, 1)

    @guppy
    @no_type_check
    def main() -> None:
        qreg = qarray(comptime(1))

        hadamard_test_pauli(qreg)
        discard_array(qreg)

    result = main.emulator(n_qubits=2).with_seed(42).with_shots(1).run()

    assert result.collated_shots()[0]["ancilla"][0] == 0


def test_make_hadamard_test_pauli_z_kickback_measures_one() -> None:
    """A ``Z`` Pauli on ``|1>`` should kick back a ``1`` ancilla outcome."""
    pauli_string = zqp.String.from_str("Z0", 1)
    hadamard_test_pauli = make_hadamard_test_pauli(pauli_string, 1)

    @guppy
    @no_type_check
    def main() -> None:
        qreg = qarray(comptime(1))

        x(qreg[0])
        hadamard_test_pauli(qreg)
        discard_array(qreg)

    result = main.emulator(n_qubits=2).with_seed(42).with_shots(1).run()

    assert result.collated_shots()[0]["ancilla"][0] == 1


def test_make_direct_measure_pauli_simple_measures_system_bitstring() -> None:
    """Measure a non-adjacent four-qubit X/Y/Z term in its eigenstate."""
    pauli_string = zqp.String.from_str("X0 Y2 Z3", 4)
    direct_measure = make_direct_measure_pauli_simple(pauli_string, 4)

    @guppy
    @no_type_check
    def main() -> None:
        qreg = qarray(comptime(4))

        h(qreg[0])
        h(qreg[2])
        s(qreg[2])
        x(qreg[3])
        direct_measure(qreg)

    result = main.emulator(n_qubits=4).with_seed(42).with_shots(8).run()

    assert [sample["bitstring"][0] for sample in result.collated_shots()] == [
        [False, False, False, True]
    ] * 8


def test_make_direct_measure_pauli_accepts_custom_four_qubit_structure() -> None:
    """Apply a custom structure before measuring the full register."""

    @guppy
    @no_type_check
    def prepare_measurement_basis(qreg: array[qubit, comptime(4)]) -> None:
        x(qreg[0])
        x(qreg[2])

    direct_measure = make_direct_measure_pauli(prepare_measurement_basis, 4)

    @guppy
    @no_type_check
    def main() -> None:
        qreg = qarray(comptime(4))
        direct_measure(qreg)

    result = main.emulator(n_qubits=4).with_seed(42).with_shots(1).run()

    assert result.collated_shots()[0]["bitstring"][0] == [True, False, True, False]
