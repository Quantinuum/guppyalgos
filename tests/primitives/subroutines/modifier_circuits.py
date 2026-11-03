"""Shared Guppy circuit for basis-state modifier tests."""

from typing import no_type_check

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


def make_modifier_circuit(
    operation: GuppyFunctionDefinition[[qubit, array[qubit, 2]], None],
    mode: str,
) -> GuppyFunctionDefinition[[bool, bool, array[bool, 2]], None]:
    """Prepare basis inputs, apply a modifier, and measure all four qubits."""

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

    return main
