"""Tests for the quantum exponentiator."""

from typing import no_type_check

import pytest

from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.quantum import collect_measurements, discard_array, measure_array, h

from guppyalgos.primitives.arithmetic import exponentiator_ripple_gidney_mod
from guppyalgos.utils import int_to_bits, qarray, apply_bitstring


@pytest.mark.parametrize(
    ("n_exponent", "n_output", "base", "exponent", "initial"),
    [
        (2, 0, 3, 3, 1),
        (2, 1, 3, 3, 1),
        (3, 2, 3, 3, 1),
        (3, 3, 3, 4, 2),
        (2, 4, 5, 2, 1),
    ],
)
def test_exponentiator_ripple_gidney_mod(
    n_exponent: int, n_output: int, base: int, exponent: int, initial: int
):
    """Test that the exponentiator correctly calculates ``base^exponent mod 2^n``."""
    exponent_bits = int_to_bits(exponent, n_exponent)
    expected = initial * (base**exponent % (2**n_output)) % (2**n_output)
    expected_output_bits = int_to_bits(expected, n_output)

    exponent_state_array = array(*exponent_bits)
    initial_state_bits = int_to_bits(initial, n_output) if n_output > 0 else []

    @guppy
    @no_type_check
    def main() -> None:
        exponent_reg = qarray(n_exponent)
        apply_bitstring(exponent_reg, exponent_state_array)
        output_reg = qarray(n_output)
        apply_bitstring(output_reg, initial_state_bits)

        exponentiator_ripple_gidney_mod(exponent_reg, output_reg, base)

        output("exponent_meas", collect_measurements(measure_array(exponent_reg)))
        if n_output > 0:
            output("output_meas", collect_measurements(measure_array(output_reg)))
        else:
            discard_array(output_reg)

    required_qubits = n_exponent + 4 * n_output + 1
    result = main.emulator(n_qubits=required_qubits).run().results[0].as_dict()
    assert result["exponent_meas"] == exponent_bits
    if n_output > 0:
        assert result["output_meas"] == expected_output_bits


@pytest.mark.parametrize(
    ("n_exponent", "n_output", "base"), [(3, 2, 3), (3, 3, 3), (5, 3, 5)]
)
def test_exponentiator_ripple_gidney_mod_superposition(
    n_exponent: int, n_output: int, base: int
) -> None:
    """Check that the exponentiator preserves a superposition on all inputs."""

    @guppy
    @no_type_check
    def main() -> None:
        exponent_reg = qarray(n_exponent)
        output_reg = qarray(n_output)

        for i in range(n_exponent):
            h(exponent_reg[i])
        for i in range(n_output):
            h(output_reg[i])

        exponentiator_ripple_gidney_mod(exponent_reg, output_reg, base)

        for i in range(n_exponent):
            h(exponent_reg[i])
        for i in range(n_output):
            h(output_reg[i])

        output("exponent_meas", collect_measurements(measure_array(exponent_reg)))
        output("output_meas", collect_measurements(measure_array(output_reg)))

    required_qubits = n_exponent + 4 * n_output + 1
    result = (
        main.emulator(n_qubits=required_qubits).with_shots(3).run().results[0].as_dict()
    )
    assert result["exponent_meas"] == [0] * n_exponent
    assert result["output_meas"] == [0] * n_output


@pytest.mark.parametrize(
    ("base", "error_message"),
    [
        (0, "requires an odd base"),
        (2, "requires an odd base"),
        (-1, "requires a non-negative base"),
        (-3, "requires a non-negative base"),
    ],
)
def test_exponentiator_rejects_invalid_classical_base(
    base: int, error_message: str
) -> None:
    """Check that the exponentiator rejects even and negative bases."""

    @guppy
    @no_type_check
    def main() -> None:
        exponent_reg = qarray(2)
        output_reg = qarray(2)
        exponentiator_ripple_gidney_mod(exponent_reg, output_reg, base)
        discard_array(exponent_reg)
        discard_array(output_reg)

    with pytest.raises(ValueError, match=error_message):
        main.emulator(n_qubits=1).run()
