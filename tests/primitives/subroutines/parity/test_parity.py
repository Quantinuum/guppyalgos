"""Tests for sequential and LAQCC parity circuits."""

from itertools import product
from typing import no_type_check

import pytest
from guppylang import guppy
from guppylang.std.builtins import array, control, dagger, comptime, output
from guppylang.std.quantum import collect_measurements, measure, measure_array, qubit, x

from guppyalgos.primitives.subroutines.parity import (
    parity_laqcc,
    parity_laqcc_total_qubits,
    parity_sequential,
)
from guppyalgos.utils import qarray


N_INPUT_QUBITS = 5


@pytest.mark.parametrize("mode", ["forward", "dagger", "control", "controlled_dagger"])
def test_parity_sequential_modifiers(mode: str) -> None:
    """Check each modifier's measured outputs for every basis input."""

    @guppy
    @no_type_check
    def main(external_bit: bool, target_bit: bool, bits: array[bool, 2]) -> None:
        external = qubit()
        target = qubit()
        data = qarray(2)
        if external_bit:
            x(external)
        if target_bit:
            x(target)
        for i in range(2):
            if bits[i]:
                x(data[i])

        if comptime(mode == "forward"):
            parity_sequential(target, data)
        elif comptime(mode == "dagger"):
            with dagger:
                parity_sequential(target, data)
        elif comptime(mode == "control"):
            with control(external):
                parity_sequential(target, data)
        else:
            with control(external), dagger:
                parity_sequential(target, data)

        output("external", measure(external).read())
        output("target", measure(target).read())
        output("data", collect_measurements(measure_array(data)))

    emulator = main.emulator(n_qubits=4).with_seed(42).with_shots(1)
    for external_bit, target_bit, data0, data1 in product([False, True], repeat=4):
        bits = [data0, data1]
        enabled = mode in ("forward", "dagger") or external_bit
        expected_data = bits
        expected_target = target_bit ^ (enabled and (data0 ^ data1))
        result = emulator.run(
            external_bit=external_bit, target_bit=target_bit, bits=bits
        ).collated_shots()[0]
        assert result["external"][0] == external_bit
        assert result["target"][0] == expected_target
        assert result["data"][0] == expected_data


@pytest.mark.parametrize(
    ("n_input_qubits", "expected_total"),
    [
        pytest.param(3, 4, id="sequential-fallback"),
        pytest.param(8, 19, id="eight-inputs"),
        pytest.param(16, 43, id="sixteen-inputs"),
    ],
)
def test_parity_laqcc_total_qubits(n_input_qubits: int, expected_total: int) -> None:
    """Count inputs, target, and LAQCC ancillas."""
    assert parity_laqcc_total_qubits(n_input_qubits) == expected_total


def _get_parity_emulator(parity, total_qubits: int):
    """Build an emulator for a parity implementation."""

    @guppy
    @no_type_check
    def main(bits: array[bool, N_INPUT_QUBITS], target_bit: bool) -> None:
        inputs = qarray(N_INPUT_QUBITS)
        target = qubit()

        for i in range(N_INPUT_QUBITS):
            if bits[i]:
                x(inputs[i])
        if target_bit:
            x(target)

        parity(target, inputs)

        output("inputs", collect_measurements(measure_array(inputs)))
        output("target", measure(target).read())

    return main.emulator(n_qubits=total_qubits).with_seed(42).with_shots(1)


@pytest.mark.parametrize(
    ("parity", "total_qubits"),
    [
        pytest.param(
            parity_sequential,
            N_INPUT_QUBITS + 1,
            id="sequential",
        ),
        pytest.param(
            parity_laqcc,
            parity_laqcc_total_qubits(N_INPUT_QUBITS),
            id="laqcc",
        ),
    ],
)
@pytest.mark.parametrize(
    ("input_bits", "target_bit", "expected_target"),
    [
        pytest.param(
            [False] * N_INPUT_QUBITS,
            False,
            False,
            id="even-input-target-zero",
        ),
        pytest.param(
            [True, *([False] * (N_INPUT_QUBITS - 1))],
            True,
            False,
            id="odd-input-target-one",
        ),
        pytest.param(
            [True, *([False] * (N_INPUT_QUBITS - 1))],
            False,
            True,
            id="odd-input-target-zero",
        ),
        pytest.param(
            [True, True, *([False] * (N_INPUT_QUBITS - 2))],
            True,
            True,
            id="even-input-target-one",
        ),
    ],
)
def test_parity_basis_states(
    parity,
    total_qubits: int,
    input_bits: list[bool],
    target_bit: bool,
    expected_target: bool,
) -> None:
    """XOR input parity into the target without changing the inputs."""
    emulator = _get_parity_emulator(parity, total_qubits)
    shots = emulator.run(bits=input_bits, target_bit=target_bit).collated_shots()

    assert shots[0]["inputs"][0] == input_bits
    assert shots[0]["target"][0] == expected_target
