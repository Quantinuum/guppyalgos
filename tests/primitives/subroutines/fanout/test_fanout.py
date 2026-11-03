"""Tests for flat basic and measurement-assisted fanout callables.

The suite checks three uses of fanout with a control prepared in ``|1>``:

* Compute copies the control into a zero-initialized target register.
* Uncompute restores a target register correlated with the control to zero.
* Parity fanout XORs the control into an arbitrary target bitstring.

The basic and parity callables support all three uses. The specialized
measurement compute and uncompute callables are each tested only against
their required target-register input state.
"""

from collections.abc import Callable
from itertools import product
from typing import no_type_check

import pytest
from guppylang import guppy
from guppylang.std.builtins import array, control, dagger, comptime, output
from guppylang.std.quantum import (
    collect_measurements,
    h,
    measure,
    measure_array,
    qubit,
    x,
)

from guppyalgos.primitives.subroutines.fanout import (
    fanout_basic,
    fanout_log,
    fanout_measurement_compute,
    fanout_measurement_compute_total_qubits,
    fanout_measurement_parity,
    fanout_measurement_parity_total_qubits,
    fanout_measurement_uncompute,
)
from guppyalgos.primitives.utilities.gates import transversal
from guppyalgos.utils import qarray


_ZERO_BITS = [False, False, False, False, False, False]
_ONE_BITS = [True, True, True, True, True, True]
_ARBITRARY_BITS = [False, True, True, False, True, False]
_XOR_BITS = [True, False, False, True, False, True]


@pytest.mark.parametrize("superposition", [False, True])
def test_fanout_basic_dagger_round_trip(superposition: bool) -> None:
    """Restore basis and superposition inputs by applying forward then dagger."""

    @guppy
    @no_type_check
    def main(target_bit: bool, bits: array[bool, 2]) -> None:
        target = qubit()
        data = qarray(2)
        if target_bit:
            x(target)
        for i in range(2):
            if bits[i]:
                x(data[i])
        if comptime(superposition):
            h(target)
            for i in range(2):
                h(data[i])

        fanout_basic(target, data)
        with dagger:
            fanout_basic(target, data)

        if comptime(superposition):
            h(target)
            for i in range(2):
                h(data[i])
        output("target", measure(target).read())
        output("data", collect_measurements(measure_array(data)))

    emulator = main.emulator(n_qubits=3).with_seed(42).with_shots(1)
    for target_bit, data0, data1 in product([False, True], repeat=3):
        bits = [data0, data1]
        result = emulator.run(target_bit=target_bit, bits=bits).collated_shots()[0]
        assert result["target"][0] == target_bit
        assert result["data"][0] == bits


@pytest.mark.parametrize(
    "daggered", [False, True], ids=["control", "controlled-dagger"]
)
@pytest.mark.parametrize("superposition", [False, True], ids=["basis", "all-plus"])
def test_fanout_basic_controlled(daggered: bool, superposition: bool) -> None:
    """Share expected outputs for controlled self-adjoint operations."""

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
        if comptime(superposition):
            h(external)
            h(target)
            for i in range(2):
                h(data[i])

        with control(external):
            if comptime(daggered):
                with dagger:
                    fanout_basic(target, data)
            else:
                fanout_basic(target, data)

        if comptime(superposition):
            h(external)
            h(target)
            for i in range(2):
                h(data[i])
        output("external", measure(external).read())
        output("target", measure(target).read())
        output("data", collect_measurements(measure_array(data)))

    emulator = main.emulator(n_qubits=4).with_seed(42).with_shots(1)
    cases = (
        [(False, False, False, False)]
        if superposition
        else product([False, True], repeat=4)
    )
    for external_bit, target_bit, data0, data1 in cases:
        bits = [data0, data1]
        expected_data = [bit ^ (external_bit and target_bit) for bit in bits]
        expected_target = target_bit
        result = emulator.run(
            external_bit=external_bit, target_bit=target_bit, bits=bits
        ).collated_shots()[0]
        assert result["external"][0] == external_bit
        assert result["target"][0] == expected_target
        assert result["data"][0] == expected_data


def _basic_total_qubits(n_target_qubits: int) -> int:
    """Count the control and target qubits."""
    return n_target_qubits + 1


@pytest.mark.parametrize(
    (
        "fanout_operation",
        "control_bit",
        "initial_bits",
        "expected_bits",
        "total_qubits_fn",
    ),
    [
        # Compute: |1>|000000> -> |1>|111111>.
        pytest.param(
            fanout_basic,
            True,
            _ZERO_BITS,
            _ONE_BITS,
            _basic_total_qubits,
            id="basic-compute",
        ),
        pytest.param(
            fanout_log,
            True,
            _ZERO_BITS,
            _ONE_BITS,
            _basic_total_qubits,
            id="log-compute",
        ),
        pytest.param(
            fanout_measurement_compute,
            True,
            _ZERO_BITS,
            _ONE_BITS,
            fanout_measurement_compute_total_qubits,
            id="measurement-compute",
        ),
        pytest.param(
            fanout_measurement_parity,
            True,
            _ZERO_BITS,
            _ONE_BITS,
            fanout_measurement_parity_total_qubits,
            id="measurement-parity-compute",
        ),
        # Uncompute: |1>|111111> -> |1>|000000>.
        pytest.param(
            fanout_basic,
            True,
            _ONE_BITS,
            _ZERO_BITS,
            _basic_total_qubits,
            id="basic-uncompute",
        ),
        pytest.param(
            fanout_log,
            True,
            _ONE_BITS,
            _ZERO_BITS,
            _basic_total_qubits,
            id="log-uncompute",
        ),
        pytest.param(
            fanout_measurement_uncompute,
            True,
            _ONE_BITS,
            _ZERO_BITS,
            _basic_total_qubits,
            id="measurement-uncompute",
        ),
        pytest.param(
            fanout_measurement_parity,
            True,
            _ONE_BITS,
            _ZERO_BITS,
            fanout_measurement_parity_total_qubits,
            id="measurement-parity-uncompute",
        ),
        # General: XOR |1> into an arbitrary target bitstring.
        pytest.param(
            fanout_basic,
            True,
            _ARBITRARY_BITS,
            _XOR_BITS,
            _basic_total_qubits,
            id="basic-general",
        ),
        pytest.param(
            fanout_log,
            True,
            _ARBITRARY_BITS,
            _XOR_BITS,
            _basic_total_qubits,
            id="log-general",
        ),
        pytest.param(
            fanout_measurement_parity,
            True,
            _ARBITRARY_BITS,
            _XOR_BITS,
            fanout_measurement_parity_total_qubits,
            id="measurement-parity",
        ),
        # A zero control leaves arbitrary targets unchanged.
        pytest.param(
            fanout_basic,
            False,
            _ARBITRARY_BITS,
            _ARBITRARY_BITS,
            _basic_total_qubits,
            id="basic-control-zero",
        ),
        pytest.param(
            fanout_log,
            False,
            _ARBITRARY_BITS,
            _ARBITRARY_BITS,
            _basic_total_qubits,
            id="log-control-zero",
        ),
        pytest.param(
            fanout_measurement_parity,
            False,
            _ARBITRARY_BITS,
            _ARBITRARY_BITS,
            fanout_measurement_parity_total_qubits,
            id="measurement-parity-control-zero",
        ),
    ],
)
def test_fanout(
    fanout_operation,
    control_bit: bool,
    initial_bits: list[bool],
    expected_bits: list[bool],
    total_qubits_fn: Callable[[int], int],
) -> None:
    """Apply a flat Guppy fanout callable directly to all target qubits.

    Each parameter row supplies the callable's valid initial target state,
    the expected state after fanout, and the qubit-count function needed by
    the emulator. Measurement-assisted implementations therefore receive
    their ancilla qubits, while basic fanout uses only the control and target
    register.

    The final assertions also verify that fanout preserves the control rather
    than consuming or changing it.
    """
    n_target_qubits = len(initial_bits)

    @guppy
    @no_type_check
    def main() -> None:
        control = qubit()
        target_qreg = qarray(n_target_qubits)
        bits = initial_bits

        if control_bit:
            x(control)
        for bit in range(n_target_qubits):
            if bits[bit]:
                x(target_qreg[bit])

        fanout_operation(control, target_qreg)

        output("control", measure(control).read())
        output("state", collect_measurements(measure_array(target_qreg)))

    result = (
        main.emulator(n_qubits=total_qubits_fn(n_target_qubits))
        .with_seed(42)
        .with_shots(1)
        .run()
        .collated_shots()[0]
    )
    assert result["state"][0] == expected_bits
    assert result["control"][0] == control_bit


@pytest.mark.parametrize(
    ("fanout_operation", "total_qubits_fn"),
    [
        pytest.param(fanout_basic, _basic_total_qubits, id="basic"),
        pytest.param(fanout_log, _basic_total_qubits, id="log"),
        pytest.param(
            fanout_measurement_parity,
            fanout_measurement_parity_total_qubits,
            id="measurement-parity",
        ),
    ],
)
def test_fanout_preserves_all_plus_state(
    fanout_operation,
    total_qubits_fn: Callable[[int], int],
) -> None:
    """Leave an all-plus state unchanged, including all relative phases."""
    n_target_qubits = len(_ZERO_BITS)

    @guppy
    @no_type_check
    def main() -> None:
        control = qubit()
        target_qreg = qarray(n_target_qubits)

        h(control)
        transversal(h, target_qreg)
        fanout_operation(control, target_qreg)
        h(control)
        transversal(h, target_qreg)

        output("control", measure(control).read())
        output("state", collect_measurements(measure_array(target_qreg)))

    result = (
        main.emulator(n_qubits=total_qubits_fn(n_target_qubits))
        .with_seed(42)
        .with_shots(1)
        .run()
        .collated_shots()[0]
    )
    assert result["state"][0] == _ZERO_BITS
    assert result["control"][0] == 0
