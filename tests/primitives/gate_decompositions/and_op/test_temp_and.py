"""Tests for temporary AND operations using different implementations."""

import pytest

from guppylang.decorator import guppy
from guppylang.std.builtins import array, control, dagger, output
from guppylang.std.quantum import (
    measure_array,
    qubit,
    toffoli,
    x,
    measure,
    collect_measurements,
    cx,
    discard_array,
    h,
)

from guppyalgos.utils import int_to_bits
from guppyalgos.utils import qarray

from guppyalgos.primitives.gate_decompositions.and_op import (
    temp_and,
)

from guppylang.defs import GuppyFunctionDefinition


from typing import no_type_check


@pytest.mark.parametrize("and_op", [toffoli, temp_and])
@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_compute_temp_and(
    i: int,
    and_op: GuppyFunctionDefinition,
) -> None:
    """Test temp_and with different implementations.

    Temporary AND which only acts on an incoming |0> target state loops
    over the incoming input bits possibilities and applies the AND operation.

    Args:
        i (int): Integer index to determine input bits.
        and_op (GuppyFunctionDefinition): The AND computation operation
            to be tested. This can be either the toffoli or the temp_and
            function.

    """
    binary_index = int_to_bits(i, 2)

    @guppy
    @no_type_check
    def main() -> None:
        idx = binary_index
        index_qreg = qarray(2)
        target = qubit()

        for bit in range(2):
            if idx[bit]:
                x(index_qreg[bit])

        and_op(index_qreg[0], index_qreg[1], target)

        output("index", collect_measurements(measure_array(index_qreg)))
        output("state", measure(target).read())

    total_qubits = 3
    my_shots = main.emulator(n_qubits=total_qubits).with_seed(42).with_shots(1).run()
    bool_and = binary_index[0] and binary_index[1]
    assert my_shots.collated_shots()[0]["state"][0] == bool_and


@pytest.mark.parametrize("and_op", [toffoli, temp_and])
@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_uncompute_temp_and(
    i: int,
    and_op: GuppyFunctionDefinition,
) -> None:
    """Test the custom dagger with different AND implementations.

    Temporary AND uncomputation acts on a target that holds the AND of the input
    bits. The test loops over the input bits possibilities and checks that the
    target is returned to |0>.

    Args:
        i (int): Integer index to determine input bits.
        and_op (GuppyFunctionDefinition): The AND computation operation
            to be tested. This can be either the toffoli or the temp_and
            function.

    """
    binary_index = int_to_bits(i, 2)

    @guppy
    @no_type_check
    def main() -> None:
        idx = binary_index
        index_qreg = qarray(2)
        target = qubit()

        for bit in range(2):
            if idx[bit]:
                x(index_qreg[bit])

        # the target holds the AND of the input bits, as the uncomputation requires
        if idx[0] and idx[1]:
            x(target)

        with dagger:
            and_op(index_qreg[0], index_qreg[1], target)

        output("index", collect_measurements(measure_array(index_qreg)))
        output("state", measure(target).read())

    total_qubits = 3
    my_shots = main.emulator(n_qubits=total_qubits).with_seed(42).with_shots(1).run()
    assert my_shots.collated_shots()[0]["state"][0] == 0


@pytest.mark.parametrize("n_controls", [0, 1, 2, 3])
@pytest.mark.parametrize("inverse", [False, True])
def test_temp_and_modifiers(n_controls: int, inverse: bool) -> None:
    """Check the promised subspace against Toffoli, including coherent controls.

    Entangling every input with a reference checks the entire AND isometry,
    including phases and input preservation. The reference Toffoli is independent
    of the custom implementation, so matching errors in compute and uncompute
    cannot cancel. Multiple shots exercise measurement-based corrections.
    """
    n_references = n_controls + 2

    # Guppy currently cannot normalize a control block over an empty array.
    if n_controls == 0:

        @guppy(unitary=True)
        @no_type_check
        def compute(
            inputs: array[qubit, 2], target: qubit, controls: array[qubit, 0]
        ) -> None:
            temp_and(inputs[0], inputs[1], target)

        @guppy
        @no_type_check
        def reference(
            inputs: array[qubit, 2], target: qubit, controls: array[qubit, 0]
        ) -> None:
            toffoli(inputs[0], inputs[1], target)

    else:

        @guppy(unitary=True)
        @no_type_check
        def compute(
            inputs: array[qubit, 2], target: qubit, controls: array[qubit, n_controls]
        ) -> None:
            with control(controls):
                temp_and(inputs[0], inputs[1], target)

        @guppy
        @no_type_check
        def reference(
            inputs: array[qubit, 2], target: qubit, controls: array[qubit, n_controls]
        ) -> None:
            # Use ordinary Toffolis so the reference is independent of modifiers.
            work = qarray(n_controls)
            toffoli(controls[0], inputs[0], work[0])
            for i in range(1, n_controls):
                toffoli(controls[i], work[i - 1], work[i])
            toffoli(work[n_controls - 1], inputs[1], target)
            for i in range(n_controls - 1, 0, -1):
                toffoli(controls[i], work[i - 1], work[i])
            toffoli(controls[0], inputs[0], work[0])
            discard_array(work)

    @guppy
    @no_type_check
    def main() -> None:
        controls = qarray(n_controls)
        inputs = qarray(2)
        references = qarray(n_references)
        target = qubit()

        for i in range(n_controls):
            h(controls[i])
            cx(controls[i], references[i])
        for i in range(2):
            h(inputs[i])
            cx(inputs[i], references[n_controls + i])

        if inverse:
            reference(inputs, target, controls)
            with dagger:
                compute(inputs, target, controls)
        else:
            compute(inputs, target, controls)
            reference(inputs, target, controls)

        for i in range(n_controls):
            cx(controls[i], references[i])
            h(controls[i])
        for i in range(2):
            cx(inputs[i], references[n_controls + i])
            h(inputs[i])

        if n_controls > 0:
            output("controls", collect_measurements(measure_array(controls)))
        else:
            discard_array(controls)
        output("inputs", collect_measurements(measure_array(inputs)))
        output("references", collect_measurements(measure_array(references)))
        output("target", measure(target).read())

    shots = main.emulator(3 * n_controls + 5).with_seed(42).with_shots(32).run()
    for shot in shots.collated_shots():
        if n_controls > 0:
            assert shot["controls"][0] == [False] * n_controls
        assert shot["inputs"][0] == [False] * 2
        assert shot["references"][0] == [False] * (n_controls + 2)
        assert shot["target"][0] == 0
