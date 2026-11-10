"""Tests for temporary AND operations using different implementations."""

import pytest

from guppylang.decorator import guppy
from guppylang.std.quantum import (
    measure_array,
    qubit,
    toffoli,
    x,
    measure,
    collect_measurements,
)

from guppyalgos.utils import int_to_bits


from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and,
)

from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and_uncomp_index,
    temp_and_comp_index,
)

from guppylang.defs import GuppyFunctionDefinition
from guppyalgos.utils import qarray

from typing import no_type_check


@pytest.mark.parametrize("and_op", [toffoli, temp_and])
@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_index_temp_and_comp(
    i: int,
    and_op: GuppyFunctionDefinition,
) -> None:
    """Test index_temp_and_comp with different implementations.

    Temporary AND which only acts on an incoming |0> target state loops
    over the incoming input bits possibilities and applies the AND operation.
    Which flips the target to |1> if both inputs are the incoming binary
    index.

    Args:
        i (int): Integer index to determine input bits.
        and_op (GuppyFunctionDefinition): The AND computation operation

    """
    binary_index = int_to_bits(i, 2)

    @guppy
    @no_type_check
    def main() -> None:
        idx = binary_index
        index_qreg = qarray(2)

        for bit in range(2):
            if idx[bit]:
                x(index_qreg[bit])

        target = temp_and_comp_index(
            index_qreg[0], idx[0], index_qreg[1], idx[1], and_op
        )

        output("index", collect_measurements(measure_array(index_qreg)))
        output("target", measure(target).read())

    total_qubits = 3
    my_shots = main.emulator(n_qubits=total_qubits).with_seed(42).with_shots(1).run()
    result = [binary_index[0], binary_index[1], 1]
    meas_index = my_shots.collated_shots()[0]["index"][0]
    meas_target = my_shots.collated_shots()[0]["target"]
    assert isinstance(meas_index, list)
    assert isinstance(meas_target, list)
    measure_array = [*meas_index, *meas_target]
    assert measure_array == result


@pytest.mark.parametrize("and_op", [toffoli, temp_and])
@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_index_uncompute_temp_and(
    i: int,
    and_op: GuppyFunctionDefinition,
) -> None:
    """Test index_temp_and_uncomp with different implementations.

    Temporary AND which acts on an incoming |1> target state loops
    over the incoming input bits possibilities and applies the AND operation.
    Which flips the target to |0> if both inputs are the incoming binary
    index.

    Args:
        i (int): Integer index to determine input bits.
        and_op (GuppyFunctionDefinition): The AND unitary whose dagger is tested,
            either toffoli or temp_and.

    """
    binary_index = int_to_bits(i, 2)

    @guppy
    @no_type_check
    def main() -> None:
        idx = binary_index
        index_qreg = qarray(2)
        target = qubit()
        x(target)

        for bit in range(2):
            if idx[bit]:
                x(index_qreg[bit])

        temp_and_uncomp_index(
            target, index_qreg[0], idx[0], index_qreg[1], idx[1], and_op
        )

        output("index", collect_measurements(measure_array(index_qreg)))

    total_qubits = 3
    my_shots = main.emulator(n_qubits=total_qubits).with_seed(42).with_shots(1).run()
    result = [binary_index[0], binary_index[1]]
    meas_index = my_shots.collated_shots()[0]["index"][0]
    assert meas_index == result
