"""Shared modifier tests for basic fanout and sequential parity."""

from collections.abc import Callable
from itertools import product
from typing import no_type_check

import pytest

from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import array, comptime, control, dagger, output
from guppylang.std.quantum import (
    collect_measurements,
    measure,
    measure_array,
    qubit,
    x,
)

from guppyalgos.utils import qarray

from guppyalgos.primitives.subroutines.fanout import fanout_basic
from guppyalgos.primitives.subroutines.parity import parity_sequential


def _fanout_output(target: bool, bits: list[bool]) -> tuple[bool, list[bool]]:
    return target, [bit ^ target for bit in bits]


def _parity_output(target: bool, bits: list[bool]) -> tuple[bool, list[bool]]:
    return target ^ bits[0] ^ bits[1], bits


@pytest.mark.parametrize(
    ("operation", "expected_output"),
    [
        pytest.param(fanout_basic, _fanout_output, id="fanout-basic"),
        pytest.param(parity_sequential, _parity_output, id="parity-sequential"),
    ],
)
@pytest.mark.parametrize("mode", ["forward", "dagger", "control", "controlled_dagger"])
def test_basic_modifiers(
    operation: GuppyFunctionDefinition[[qubit, array[qubit, 2]], None],
    expected_output: Callable[[bool, list[bool]], tuple[bool, list[bool]]],
    mode: str,
) -> None:
    """Check each primitive's modifier outputs for every basis input."""

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
            operation(target, data)
        elif comptime(mode == "dagger"):
            with dagger:
                operation(target, data)
        elif comptime(mode == "control"):
            with control(external):
                operation(target, data)
        else:
            with control(external), dagger:
                operation(target, data)

        output("external", measure(external).read())
        output("target", measure(target).read())
        output("data", collect_measurements(measure_array(data)))

    emulator = main.emulator(n_qubits=4).with_seed(42).with_shots(1)
    for external_bit, target_bit, data0, data1 in product([False, True], repeat=4):
        bits = [data0, data1]
        enabled = mode in ("forward", "dagger") or external_bit
        expected_target, expected_data = (
            expected_output(target_bit, bits) if enabled else (target_bit, bits)
        )
        result = emulator.run(
            external_bit=external_bit, target_bit=target_bit, bits=bits
        ).collated_shots()[0]
        assert result["external"][0] == external_bit
        assert result["target"][0] == expected_target
        assert result["data"][0] == expected_data
